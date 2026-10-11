//! fearless_simd-based rewrite of the codec hot path (experiment branch).
//!
//! Architecture mirrors the intrinsics backends in `simd.rs`, but every load,
//! store and lane op goes through the safe portable API:
//!
//! * `count_sub_ge`: fully portable generic kernel over `S: Simd` (u8x64 fixed-width type; narrower
//!   levels emulate it).
//! * encode: 64-byte kernel at the AVX-512 level with portable lane ops; two raw drops per kernel
//!   family (see the `kernel!` items): byte compaction (`vpcompressb`) and the escape mask's
//!   odd-bit spread (`pdep`; other arches fall back to the portable Morton expansion). 16-byte
//!   kernel elsewhere / for tails: the SSSE3 quarter-LUT algorithm via `swizzle_dyn_within_blocks`.
//! * decode: portable 16-byte kernels (pshufb-LUT compaction) at every level, plus a 64-byte kernel
//!   for the AVX-512 level that keeps the intrinsics backend's u64-mask leader/follower recurrence
//!   and reconstructs escapes in u8 lanes; compaction again `compress_b512`. The portable API has
//!   no byte-compress / bit-spread op, hence the two `kernel!` escape hatches (the Avx512 token's
//!   feature set covers them). chains; `vectorize()`/`#[inline(always)]` provide the target-feature
//!   contexts instead of `#[target_feature]` + `unsafe`.
//!
//! The remaining `unsafe` is the raw-pointer→slice view for variable-offset
//! stores into the caller-reserved Vec capacity (same contract as `simd.rs`).

// kernel helpers must inline into the `vectorize`/`dispatch!` target-feature
// contexts so the lane ops fuse into the caller's loop
#![allow(
    clippy::inline_always,
    reason = "splitting a kernel from its feature context adds a call boundary per block"
)]

#[cfg(target_arch = "x86_64")]
use fearless_simd::Avx512;
use fearless_simd::{Level, dispatch, mask8x16, mask8x64, prelude::*, u8x16, u8x64};
#[cfg(target_arch = "x86_64")]
use {
    super::simd::compact::COMPACT,
    core::arch::x86_64::{__mmask64, _mm512_maskz_compress_epi8, _pdep_u64},
};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::simd::compact::SPREAD4;
use super::simd::scalar;

// `is_x86_feature_detected!` results must be cached: `Level::new()` runs 11
// CPUID-backed checks, and the codec entry points hit this per call.
fn cached_level() -> Level {
    static LEVEL: std::sync::OnceLock<Level> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(Level::new)
}

// ---- count (encode_len) ----------------------------------------------------

/// Counts bytes where `b.wrapping_sub(sub) >= threshold` (`threshold >= 1`).
pub(crate) fn count_sub_ge<const KF: bool>(src: &[u8], sub: u8, threshold: u8) -> usize {
    // Experiment: hand-rolled `#[target_feature]` wrapper around the generic
    // kernel instead of `dispatch!`/`vectorize`. The wrapper's feature
    // context lets every portable op inline and fuse into one loop, without
    // the macro's closure crossing a non-inlinable boundary (closure env via
    // memory, ~3 cycles/call on small inputs). Cost: the dispatch safety is
    // now this crate's job (caller must match the detected level).
    #[cfg(target_arch = "x86_64")]
    {
        // Feature set must be a superset of every op's `kernel!` list
        // (the full Ice Lake set), or LLVM refuses to inline the ops into
        // this context and the kernel de-fuses.
        #[target_feature(enable = "adx,aes,avx512bitalg,avx512bw,avx512cd,avx512dq,avx512f,avx512ifma,avx512vbmi,avx512vbmi2,avx512vl,avx512vnni,avx512vpopcntdq,bmi1,bmi2,cmpxchg16b,fma,fxsr,gfni,lzcnt,movbe,pclmulqdq,popcnt,rdrand,rdseed,sha,vaes,vpclmulqdq,xsave,xsavec,xsaveopt,xsaves")]
        unsafe fn count512<const KF: bool>(
            avx: Avx512,
            src: &[u8],
            sub: u8,
            threshold: u8,
        ) -> usize {
            count_kernel::<KF, _>(avx, src, sub, threshold)
        }
        if let Level::Avx512(avx) = cached_level() {
            // SAFETY: level runtime-detected above; the wrapper's feature
            // set covers everything count_kernel instantiates at this level.
            return unsafe { count512::<KF>(avx, src, sub, threshold) };
        }
    }
    dispatch!(
        cached_level(),
        simd => count_kernel::<KF, _>(simd, src, sub, threshold)
    )
}

#[inline(always)]
fn count_kernel<const KF: bool, S: Simd>(simd: S, src: &[u8], sub: u8, threshold: u8) -> usize {
    let vt = u8x64::splat(simd, threshold);
    let vs = u8x64::splat(simd, sub);
    let (chunks, remainder) = src.as_chunks::<64>();
    let mut acc = 0u32;
    for c in chunks {
        let v = u8x64::from_slice(simd, c);
        let v = if KF {
            v - vs
        } else {
            v
        };
        acc += simd.simd_ge_u8x64(v, vt).to_bitmask().count_ones();
    }
    acc as usize + scalar::count_sub_ge::<KF>(remainder, sub, threshold)
}

// ---- shared helpers --------------------------------------------------------

// SAFETY (every store below): `dst[pos..pos+N]` lies inside the capacity the
// caller reserved (`2*src.len()+64` encode, `src.len()+64` decode); the bound
// proofs sit next to each kernel. Padding bytes past the cursor are
// overwritten by later stores or cut by the caller's final `set_len`.
#[inline(always)]
fn store64_at<S: Simd>(dst: *mut u8, pos: usize, v: u8x64<S>) {
    v.store_slice(unsafe { core::slice::from_raw_parts_mut(dst.add(pos), 64) });
}

#[inline(always)]
fn store16_at<S: Simd>(dst: *mut u8, pos: usize, v: u8x16<S>) {
    v.store_slice(unsafe { core::slice::from_raw_parts_mut(dst.add(pos), 16) });
}

// ---- encode ----------------------------------------------------------------

/// Encodes whole 64-byte blocks; returns the consumed prefix of `src`.
/// Same store-capacity contract as `simd::encode_simd`.
pub(crate) fn encode_simd<const KF: bool>(
    dst_ptr: *mut u8,
    p: &mut usize,
    src: &[u8],
    kf8: u8,
) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        let level = cached_level();
        if let Some(avx) = level.as_avx512() {
            // The 16..63-byte tail runs through the 16-byte kernel.
            return avx.vectorize(
                #[inline(always)]
                || {
                    let consumed = encode64::<KF>(avx, dst_ptr, p, src, kf8);
                    consumed + encode16::<KF, _>(avx, dst_ptr, p, &src[consumed..], kf8)
                },
            );
        }
        // The quarter-LUT compaction needs zero-filling byte shuffles;
        // SSE4.2 is the lowest x86 level guaranteeing them (see decode16).
        if let Some(sse42) = level.as_sse4_2() {
            return sse42.vectorize(
                #[inline(always)]
                || encode16::<KF, _>(sse42, dst_ptr, p, src, kf8),
            );
        }
        0
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is baseline; vqtbl1 zero-fills like pshufb.
        if let Some(neon) = cached_level().as_neon() {
            return neon.vectorize(
                #[inline(always)]
                || encode16::<KF, _>(neon, dst_ptr, p, src, kf8),
            );
        }
        0
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (dst_ptr, p, src, kf8);
        0
    }
}

/// 64-byte encode kernel for the AVX-512 level: portable lane ops plus the
/// `compress_b512` drop (non-escape follower lanes sit *between* kept
/// lanes, so real compaction is mandatory — a full store would leave gaps).
/// The odd-bit spread of the escape mask uses the `pdep` kernel drop; the
/// portable Morton expansion (`spread_odd`) costs ~15 scalar ops per half
/// and measurably regressed encode.
///
/// Store bounds: the two compaction stores cover `[2*off, 2*off+128)` at
/// most (with `off <= src.len()-64`), inside the reserved
/// `2*src.len()+64` bytes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn encode64<const KF: bool>(
    avx: Avx512,
    dst_ptr: *mut u8,
    p: &mut usize,
    src: &[u8],
    kf8: u8,
) -> usize {
    const EVEN64: u64 = 0x5555_5555_5555_5555;
    let (chunks, _) = src.as_chunks::<64>();
    let c93 = u8x64::splat(avx, 93);
    let c186 = u8x64::splat(avx, 186);
    let c20 = u8x64::splat(avx, 0x20);
    let c2_base = u8x64::splat(avx, 0x20u8.wrapping_sub(93));
    let c7d = u8x64::splat(avx, 0x7d);
    let c7e = u8x64::splat(avx, 0x7e);
    let kf_vec = u8x64::splat(avx, kf8);
    for c in chunks {
        let b = u8x64::from_slice(avx, c);
        let v = if KF {
            b - kf_vec
        } else {
            b
        };
        let esc = avx.simd_ge_u8x64(v, c93);
        let hi = avx.simd_ge_u8x64(v, c186);

        // c1 = esc ? 0x7D + (v >= 186) : 0x20 + v
        let c1 = esc.select(hi.select(c7e, c7d), v + c20);
        // c2 = 0x20 + v - 93 - 93*(v >= 186), wrapping (only used on esc)
        let c2 = (v + c2_base) - hi.select(c93, u8x64::splat(avx, 0));

        // Interleave (c1_i, c2_i) pairs; halves cover inputs 0..32 / 32..64.
        let z0 = avx.zip_low_u8x64(c1, c2);
        let z1 = avx.zip_high_u8x64(c1, c2);
        let esc_bits = esc.to_bitmask();
        // Keep mask: even lanes always (leaders), odd lanes on escape.
        let keep0 = EVEN64 | spread_bits(avx, esc_bits as u32);
        let keep1 = EVEN64 | spread_bits(avx, (esc_bits >> 32) as u32);
        store64_at(dst_ptr, *p, compress_b512(avx, z0, keep0));
        *p += 32 + (esc_bits as u32).count_ones() as usize;
        store64_at(dst_ptr, *p, compress_b512(avx, z1, keep1));
        *p += 32 + (esc_bits >> 32).count_ones() as usize;
    }
    chunks.len() * 64
}

/// Spreads bit i of `x` to bit 2i+1 of the result: the portable
/// (pdep-free) odd-bit expansion used by the 64-byte encode kernel.
#[cfg(target_arch = "x86_64")]
/// `spread_odd` via `pdep` inside the AVX-512 feature context.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn spread_bits(avx: Avx512, x: u32) -> u64 {
    let _ = avx;
    spread_odd_pdep(avx, x)
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn spread_bits(_avx: (), x: u32) -> u64 {
    spread_odd(x)
}

#[cfg(not(target_arch = "x86_64"))]
const fn spread_odd(x: u32) -> u64 {
    let t = x as u64;
    let t = (t | (t << 16)) & 0x0000_ffff_0000_ffff;
    let t = (t | (t << 8)) & 0x00ff_00ff_00ff_00ff;
    let t = (t | (t << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    let t = (t | (t << 2)) & 0x3333_3333_3333_3333;
    let t = (t | (t << 1)) & 0x5555_5555_5555_5555;
    t << 1
}

/// Portable 16-byte encode kernel (tail on AVX-512, full path elsewhere):
/// per-quarter LUT compaction through `swizzle_dyn_within_blocks`, the
/// SSSE3 kernel's algorithm. Requires zero-filling byte shuffles (see
/// `decode16`).
///
/// Store bounds: the four quarter stores per block cover `[2*off,
/// 2*off+40)` at most (with `off <= src.len()-16`), inside the reserved
/// `2*src.len()+64` bytes.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn encode16<const KF: bool, S: Simd>(
    simd: S,
    dst_ptr: *mut u8,
    p: &mut usize,
    src: &[u8],
    kf8: u8,
) -> usize {
    let (chunks, _) = src.as_chunks::<16>();
    let c93 = u8x16::splat(simd, 93);
    let c186 = u8x16::splat(simd, 186);
    let c20 = u8x16::splat(simd, 0x20);
    let c2_base = u8x16::splat(simd, 0x20u8.wrapping_sub(93));
    let c7d = u8x16::splat(simd, 0x7d);
    let c7e = u8x16::splat(simd, 0x7e);
    let zero = u8x16::splat(simd, 0);
    let kf_vec = u8x16::splat(simd, kf8);
    for c in chunks {
        let b = u8x16::from_slice(simd, c);
        let v = if KF {
            b - kf_vec
        } else {
            b
        };
        let esc = simd.simd_ge_u8x16(v, c93);
        let hi = simd.simd_ge_u8x16(v, c186);
        let c1 = esc.select(hi.select(c7e, c7d), v + c20);
        let c2 = (v + c2_base) - hi.select(c93, zero);
        let z0 = simd.zip_low_u8x16(c1, c2);
        let z1 = simd.zip_high_u8x16(c1, c2);

        // Per-quarter compaction: quarter q covers scratch slots 8q..8q+8
        // (inputs 4q..4q+4); its delete mask has bit 2t+1 set when input
        // 4q+t did not escape. Odd quarters read the high 8 lanes of their
        // half (`slide_within_blocks::<8>` zero-extends like `_mm_loadl`).
        let m = esc.to_bitmask() as u16;
        let halves = [z0, z1];
        for q in 0..4usize {
            let nib = ((m >> (4 * q)) & 0xf) as usize;
            // spread nib to even bits, shift to odd slots: the keep mask is
            // 0x55 | (spread << 1); the delete mask is its complement.
            let keep = 0x55u8 | ((SPREAD4[nib] << 1) & 0xaa);
            let del = keep ^ 0xff;
            let row =
                u8x16::from_slice(simd, &COMPACT.0[del as usize * 16..del as usize * 16 + 16]);
            let v8 = if q % 2 == 0 {
                halves[q / 2]
            } else {
                halves[q / 2].slide_within_blocks::<8>(zero)
            };
            let packed = simd.swizzle_dyn_within_blocks_u8x16(v8, row);
            let cnt = COMPACT.1[del as usize] as usize;
            store16_at(dst_ptr, *p, packed);
            *p += cnt;
        }
    }
    chunks.len() * 16
}

// ---- decode ----------------------------------------------------------------

// The only raw intrinsic in the rewrite: `vpcompressb` has no portable
// counterpart. `kernel!` grants the Avx512 target-feature context (Ice Lake
// set incl. VBMI2), so the call is safe.
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline(always)]
    fn compress_b512(avx: Avx512, v: u8x64<Avx512>, keep: u64) -> u8x64<Avx512> {
        _mm512_maskz_compress_epi8(keep as __mmask64, v.into()).simd_into(avx)
    }
);

// Second raw drop: spreading the escape mask onto odd bit positions is a
// bit-level permutation with no portable counterpart; BMI2 `pdep` does it in
// one instruction (the portable Morton expansion costs ~15 ops per half).
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline(always)]
    fn spread_odd_pdep(avx: Avx512, x: u32) -> u64 {
        // 32 input bits need 32 mask bits: a 64-bit pdep, not `_pdep_u32`
        _pdep_u64(u64::from(x), 0x5555_5555_5555_5555) << 1
    }
);

/// Decodes whole blocks; semantics, roll-back and hand-off invariants are
/// identical to `simd::decode_simd` (which see).
pub(crate) fn decode_simd<const KF: bool>(
    dst_ptr: *mut u8,
    p: &mut usize,
    src: &[u8],
    kf8: u8,
) -> Result<(usize, usize), (usize, usize)> {
    #[cfg(target_arch = "x86_64")]
    {
        let level = cached_level();
        if let Some(avx) = level.as_avx512() {
            return avx.vectorize(
                #[inline(always)]
                || decode512_chain::<KF>(avx, dst_ptr, p, src, kf8),
            );
        }
        // pshufb/vqtbl1 zero-fill of deleted lanes is required by the LUT;
        // SSE4.2 is the lowest x86 level guaranteeing it.
        if let Some(sse42) = level.as_sse4_2() {
            return sse42.vectorize(
                #[inline(always)]
                || decode16::<KF, _>(sse42, dst_ptr, p, src, kf8),
            );
        }
        Ok((0, 0))
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is baseline; vqtbl1 zero-fills like pshufb.
        if let Some(neon) = cached_level().as_neon() {
            return neon.vectorize(
                #[inline(always)]
                || decode16::<KF, _>(neon, dst_ptr, p, src, kf8),
            );
        }
        Ok((0, 0))
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (dst_ptr, p, src, kf8);
        Ok((0, 0))
    }
}

/// AVX-512 chain: 64-byte kernel, then the 16-byte kernel for the 16..63
/// tail. On success no cross-block pair is pending (see `simd::decode_simd`).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn decode512_chain<const KF: bool>(
    avx: Avx512,
    dst_ptr: *mut u8,
    p: &mut usize,
    src: &[u8],
    kf8: u8,
) -> Result<(usize, usize), (usize, usize)> {
    let consumed = match decode512::<KF>(avx, dst_ptr, p, src, kf8) {
        Ok((c, _)) => c,
        Err(e) => return Err(e),
    };
    if src.len() - consumed >= 16 {
        return match decode16::<KF, _>(avx, dst_ptr, p, &src[consumed..], kf8) {
            Ok((c2, _)) => Ok((consumed + c2, *p)),
            Err((c2, _)) => Err((consumed + c2, *p)),
        };
    }
    Ok((consumed, *p))
}

/// 64-byte decode kernel for the AVX-512 level: the intrinsics backend's
/// algorithm with portable lane ops; compaction is the `compress_b512` drop.
///
/// Store bounds: each block stores 64 bytes at `*p <= src.len() - produced`,
/// inside the reserved `src.len()+64` capacity.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn decode512<const KF: bool>(
    avx: Avx512,
    dst_ptr: *mut u8,
    pos: &mut usize,
    src: &[u8],
    kf8: u8,
) -> Result<(usize, usize), (usize, usize)> {
    let n_blocks = src.len() / 64;
    if n_blocks == 0 {
        return Ok((0, 0));
    }
    // vpermb control: lane i <- lane i-1, lane 0 patched with the previous
    // block's last char afterwards.
    let shift_idx = u8x64::load_array(
        avx,
        core::array::from_fn(|i| if i == 0 { 0 } else { i - 1 } as u8),
    );
    let c7d = u8x64::splat(avx, 0x7d);
    let c7e = u8x64::splat(avx, 0x7e);
    let c20 = u8x64::splat(avx, 0x20);
    let c69 = u8x64::splat(avx, 69);
    let c93 = u8x64::splat(avx, 93);
    let c186 = u8x64::splat(avx, 186);
    let kf_vec = u8x64::splat(avx, kf8);
    let lane0 = mask8x64::from_bitmask(avx, 1);

    let mut off = 0usize;
    let mut carry = false;
    let mut prev_char = 0u8;
    let mut safe = (0usize, 0usize);
    // keep the write cursor in a register; `*pos` is written back only on the
    // exit paths (writing through the reference inside the loop forces a
    // per-block reload+store of the destination pointer)
    let mut pos_v = *pos;
    // iterate pre-checked chunks: pointer loads only, no per-block bounds
    // checks (a `while` loop with `&src[off..off+64]` costs 2 branches/block)
    let (chunks, _) = src.as_chunks::<64>();
    for chunk in chunks {
        if !carry {
            safe = (off, pos_v);
        }
        let v = u8x64::from_slice(avx, chunk);
        let mut r = avx.simd_ge_u8x64(v, c7d).to_bitmask();
        if carry {
            r &= !1;
        }
        // Runs of 7+ raw leaders are left to the scalar path (see the
        // SSSE3 kernel); the recurrence below settles runs <= 6.
        let long_run = r & (r << 1) & (r << 2) & (r << 3) & (r << 4) & (r << 5) & (r << 6);
        if long_run != 0 {
            *pos = safe.1;
            return Err(safe);
        }
        // Effective leaders: e[i] = r[i] & !e[i-1].
        let mut e = r & !(r << 1);
        e = r & !(e << 1);
        e = r & !(e << 1);
        e = r & !(e << 1);
        e = r & !(e << 1);
        e = r & !(e << 1);
        let fmask = mask8x64::from_bitmask(avx, (e << 1) | u64::from(carry));

        let vl = lane0.select(
            u8x64::splat(avx, prev_char),
            avx.swizzle_dyn_u8x64(v, shift_idx),
        );
        let b = v - c20;
        // ev = b + 93 + 93*(prev == 0x7E); garbage on non-follower lanes is
        // blended away (a wrapped sum there is harmless). Overflow past 0xFF
        // happens exactly when prev == 0x7E && b > 69.
        let q2 = avx.simd_eq_u8x64(vl, c7e);
        let ev8 = q2.select(b + c186, b + c93);
        let ovf = q2 & avx.simd_gt_u8x64(b, c69);

        let bad = (avx.simd_gt_u8x64(v, c7e) // leader > 0x7E
            | avx.simd_lt_u8x64(v, c20) // char < 0x20
            | ((ovf | avx.simd_gt_u8x64(v, c7d)) & fmask)) // bad escape/follower
            .to_bitmask();
        if bad != 0 {
            *pos = safe.1;
            return Err(safe);
        }

        // val = follower ? esc_val : (c-0x20), plus kf (wrapping).
        let mut val = fmask.select(ev8, b);
        if KF {
            val += kf_vec;
        }

        // Compact out the effective-leader lanes in one instruction.
        let produced = 64 - e.count_ones() as usize;
        store64_at(dst_ptr, pos_v, compress_b512(avx, val, !e));
        pos_v += produced;
        off += 64;
        // chunk[63]: compile-time in-bounds index; `src[off - 1]` kept a
        // per-block bounds-check + panic edge in the loop.
        prev_char = chunk[63];
        carry = e & (1 << 63) != 0;

        // Consume a trailing pair whose follower lies in the scalar tail
        // (see the SSSE3 kernel).
        if carry && off == n_blocks * 64 {
            if let Some(c2) = src.get(off).copied() {
                let q15 = u32::from(prev_char) - 0x20 - 92;
                // Wrapping sub: c2 < 0x20 wraps past 93 and is rejected by
                // the same check.
                let b2 = u32::from(c2).wrapping_sub(0x20);
                let v_esc = q15.wrapping_mul(93).wrapping_add(b2);
                if !(1..=2).contains(&q15) || b2 > 93 || v_esc > 0xff {
                    *pos = safe.1;
                    return Err(safe);
                }
                unsafe {
                    *dst_ptr.add(pos_v) = if KF {
                        (v_esc as u8).wrapping_add(kf8)
                    } else {
                        v_esc as u8
                    };
                }
                pos_v += 1;
                off += 1;
                carry = false;
            } else {
                *pos = safe.1;
                return Err(safe);
            }
        }
    }
    *pos = pos_v;
    Ok((off, pos_v))
}

/// Portable 16-byte decode kernel: the SSSE3 algorithm with the leader
/// recurrence on scalar bitmasks and escape reconstruction in u8 lanes.
///
/// Requires zero-filling byte shuffles (SSE4.2+ / NEON): LUT rows mark
/// deleted lanes with 0x80, which pshufb/vqtbl1 turn into zero bytes.
#[inline(always)]
fn decode16<const KF: bool, S: Simd>(
    simd: S,
    dst_ptr: *mut u8,
    pos: &mut usize,
    src: &[u8],
    kf8: u8,
) -> Result<(usize, usize), (usize, usize)> {
    let n_blocks = src.len() / 16;
    if n_blocks == 0 {
        return Ok((0, 0));
    }
    let c7d = u8x16::splat(simd, 0x7d);
    let c7e = u8x16::splat(simd, 0x7e);
    let c20 = u8x16::splat(simd, 0x20);
    let c69 = u8x16::splat(simd, 69);
    let c93 = u8x16::splat(simd, 93);
    let c186 = u8x16::splat(simd, 186);
    let kf_vec = u8x16::splat(simd, kf8);
    let zero = u8x16::splat(simd, 0);
    let lane0 = mask8x16::from_bitmask(simd, 1);

    let mut off = 0usize;
    let mut carry = false;
    let mut prev_char = 0u8;
    let mut safe = (0usize, 0usize);
    let mut block = 0usize;
    while block < n_blocks {
        if !carry {
            safe = (off, *pos);
        }
        let v = u8x16::from_slice(simd, &src[off..off + 16]);
        let mut r = simd.simd_ge_u8x16(v, c7d).to_bitmask() as u16;
        if carry {
            r &= !1;
        }
        let long_run = r & (r << 1) & (r << 2) & (r << 3) & (r << 4) & (r << 5) & (r << 6);
        if long_run != 0 {
            *pos = safe.1;
            return Err(safe);
        }
        // Effective leaders: e[i] = r[i] & !e[i-1] (settles runs <= 6).
        let mut e = r & !(r << 1);
        e = r & !(e << 1);
        e = r & !(e << 1);
        e = r & !(e << 1);
        e = r & !(e << 1);
        let fmask = mask8x16::from_bitmask(simd, u64::from((e << 1) | u16::from(carry)));

        // lane i <- v[i-1], lane 0 <- prev char.
        let vl = lane0.select(
            u8x16::splat(simd, prev_char),
            zero.slide_within_blocks::<15>(v),
        );
        let b = v - c20;
        let q2 = simd.simd_eq_u8x16(vl, c7e);
        let ev8 = q2.select(b + c186, b + c93);
        let ovf = q2 & simd.simd_gt_u8x16(b, c69);

        let bad = (simd.simd_gt_u8x16(v, c7e)
            | simd.simd_lt_u8x16(v, c20)
            | ((ovf | simd.simd_gt_u8x16(v, c7d)) & fmask))
            .to_bitmask();
        if bad != 0 {
            *pos = safe.1;
            return Err(safe);
        }

        let mut val = fmask.select(ev8, b);
        if KF {
            val += kf_vec;
        }

        // Compact out the leader lanes via the shared LUT (see `compact`):
        // a row's low half packs lanes 0..8 to result 0..8; the shared rows
        // also carry the +8-offset indices for lanes 8..16 in their high
        // half, so the second shuffle's result only needs sliding down 8
        // bytes (the baseline's `_mm_unpackhi_epi64`). Junk tails of both
        // 16-byte stores are overwritten by the next store or cut by the
        // caller's `set_len`.
        let lo = (e & 0xff) as usize;
        let hi = (e >> 8) as usize;
        let row_lo = u8x16::from_slice(simd, &COMPACT.0[lo * 16..lo * 16 + 16]);
        let row_hi = u8x16::from_slice(simd, &COMPACT.0[hi * 16..hi * 16 + 16]);
        let lo_packed = simd.swizzle_dyn_within_blocks_u8x16(val, row_lo);
        let hi_packed = simd
            .swizzle_dyn_within_blocks_u8x16(val, row_hi)
            .slide_within_blocks::<8>(zero);
        store16_at(dst_ptr, *pos, lo_packed);
        *pos += COMPACT.1[lo] as usize;
        store16_at(dst_ptr, *pos, hi_packed);
        *pos += COMPACT.1[256 + hi] as usize;
        off += 16;
        prev_char = src[off - 1];
        carry = e & 0x8000 != 0;
        block += 1;

        // Trailing pair consumption (see the SSSE3 kernel).
        if carry && block == n_blocks {
            if let Some(c2) = src.get(off).copied() {
                let q15 = u32::from(prev_char) - 0x20 - 92;
                let b2 = u32::from(c2).wrapping_sub(0x20);
                let v_esc = q15.wrapping_mul(93).wrapping_add(b2);
                if !(1..=2).contains(&q15) || b2 > 93 || v_esc > 0xff {
                    *pos = safe.1;
                    return Err(safe);
                }
                unsafe {
                    *dst_ptr.add(*pos) = if KF {
                        (v_esc as u8).wrapping_add(kf8)
                    } else {
                        v_esc as u8
                    };
                }
                *pos += 1;
                off += 1;
                carry = false;
            } else {
                *pos = safe.1;
                return Err(safe);
            }
        }
    }
    Ok((off, *pos))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        let mut s = 0x0bad_c0de_dead_beefu64;
        for b in &mut v {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *b = (s >> 56) as u8;
        }
        v
    }

    #[test]
    fn fearless_matches_scalar() {
        for n in [
            0usize, 1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 1024, 65539,
        ] {
            let src = sample(n);
            for kf in [0u8, 1, 0x5a, 0xa7] {
                assert_eq!(
                    count_sub_ge::<true>(&src, kf, 93),
                    scalar::count_sub_ge::<true>(&src, kf, 93),
                    "count n={n} kf={kf}"
                );
                assert_eq!(
                    count_sub_ge::<false>(&src, 0, 93),
                    scalar::count_sub_ge::<false>(&src, 0, 93),
                    "count-plain n={n}"
                );
            }
        }
    }

    /// Encode/decode equivalence against the intrinsics backend over random
    /// and adversarial shapes (drives every kernel and tail combination).
    #[test]
    fn fearless_matches_intrinsics_end_to_end() {
        let mut datasets = vec![sample(65539), vec![0x7du8; 4096], vec![7u8; 4095]];
        datasets.push((0..8192u32).map(|i| (i % 256) as u8).collect());
        for data in datasets {
            for kf8 in [0u8, 0x5a] {
                for kf in [false, true] {
                    let kf_val = if kf {
                        kf8
                    } else {
                        0
                    };
                    let mut a = Vec::new();
                    let mut b = Vec::new();
                    let mut pa = 0usize;
                    let mut pb = 0usize;
                    let cap_a = 2 * data.len() + 64;
                    let cap_b = 2 * data.len() + 64;
                    a.resize(cap_a, 0);
                    b.resize(cap_b, 0);
                    let ca = if kf {
                        super::super::simd::encode_simd::<true>(a.as_mut_ptr(), &mut pa, &data, kf8)
                    } else {
                        super::super::simd::encode_simd::<false>(a.as_mut_ptr(), &mut pa, &data, 0)
                    };
                    let cb = if kf {
                        encode_simd::<true>(b.as_mut_ptr(), &mut pb, &data, kf8)
                    } else {
                        encode_simd::<false>(b.as_mut_ptr(), &mut pb, &data, 0)
                    };
                    assert_eq!(ca, cb, "encode consumed kf={kf_val}");
                    assert_eq!(&a[..pa], &b[..pb], "encode bytes kf={kf_val}");

                    // decode: feed each backend its own encoded output
                    let mut da2 = vec![0u8; pa + 64];
                    let mut db2 = vec![0u8; pb + 64];
                    let mut qa = 0usize;
                    let mut qb = 0usize;
                    let ra2 = if kf {
                        crate::simd::decode_simd::<true>(da2.as_mut_ptr(), &mut qa, &a[..pa], kf8)
                    } else {
                        crate::simd::decode_simd::<false>(da2.as_mut_ptr(), &mut qa, &a[..pa], 0)
                    };
                    let rb2 = if kf {
                        decode_simd::<true>(db2.as_mut_ptr(), &mut qb, &b[..pb], kf8)
                    } else {
                        decode_simd::<false>(db2.as_mut_ptr(), &mut qb, &b[..pb], 0)
                    };
                    let _ = qb;
                    match (&ra2, &rb2) {
                        (Ok(x), Ok(y)) => {
                            assert_eq!(x, y, "decode consumed kf={kf_val}");
                            assert_eq!(&da2[..x.1], &db2[..y.1], "decode bytes kf={kf_val}");
                        },
                        (Err(x), Err(y)) => {
                            assert_eq!(x, y, "decode err kf={kf_val}");
                        },
                        _ => panic!("ok-ness mismatch kf={kf_val}"),
                    }
                }
            }
        }
    }
}
