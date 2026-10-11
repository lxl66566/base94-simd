//! Deterministic instruction-count A/B for the two SIMD backends
//! (`fearless` feature). Wall-clock benches live in `codec.rs`; this file
//! counts executed instructions under Valgrind, immune to CPU frequency
//! drift.
//!
//! Caveat: valgrind's CPUID emulation does not report AVX-512 (checked
//! 3.27), so on AVX-512 hosts these numbers exercise the fallback paths
//! (intrinsics: SSE2 kernel; fearless: generic emulation at the Sse4_2
//! level) — useful for regression-pinning those, not for arbitrating the
//! AVX-512 kernels.
//!
//! Run: `GUNGRAUN_RUNNER=$(which gungraun-runner) cargo bench --features
//! fearless --bench gungraun_ab` (the env var is read at compile time by
//! `option_env!`; touch the file if you flip it).
//!
//! decode benches include sample+encode setup symmetrically; subtract the
//! matching `encode_*` numbers to isolate the decode kernel.

// gungraun's macro expansion emits fully qualified paths into the wrapper
// modules; the crate-level unused_qualifications lint fires on them.
#![allow(unused_qualifications)]

use std::hint::black_box;

use gungraun::prelude::*;

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

fn encoded(n: usize) -> Vec<u8> {
    let mut e = Vec::new();
    base94_simd::encode_into(&mut e, &sample(n));
    e
}

#[library_benchmark]
#[bench::small(64)]
#[bench::medium(1024)]
#[bench::large(16384)]
fn len_intr(n: usize) -> usize {
    let d = sample(n);
    black_box(base94_simd::ab::encoded_len_intr(black_box(&d)))
}

#[library_benchmark]
#[bench::small(64)]
#[bench::medium(1024)]
#[bench::large(16384)]
fn len_fear(n: usize) -> usize {
    let d = sample(n);
    black_box(base94_simd::ab::encoded_len_fear(black_box(&d)))
}

#[library_benchmark]
#[bench::small(64)]
#[bench::medium(1024)]
#[bench::large(16384)]
fn encode_intr(n: usize) -> usize {
    let d = sample(n);
    let mut out = Vec::new();
    base94_simd::ab::encode_into_intr(black_box(&mut out), black_box(&d));
    out.len()
}

#[library_benchmark]
#[bench::small(64)]
#[bench::medium(1024)]
#[bench::large(16384)]
fn encode_fear(n: usize) -> usize {
    let d = sample(n);
    let mut out = Vec::new();
    base94_simd::ab::encode_into_fear(black_box(&mut out), black_box(&d));
    out.len()
}

#[library_benchmark]
#[bench::small(64)]
#[bench::medium(1024)]
#[bench::large(16384)]
fn decode_intr(n: usize) -> usize {
    let e = encoded(n);
    let mut out = Vec::new();
    base94_simd::ab::decode_into_intr(black_box(&mut out), black_box(&e)).unwrap();
    out.len()
}

#[library_benchmark]
#[bench::small(64)]
#[bench::medium(1024)]
#[bench::large(16384)]
fn decode_fear(n: usize) -> usize {
    let e = encoded(n);
    let mut out = Vec::new();
    base94_simd::ab::decode_into_fear(black_box(&mut out), black_box(&e)).unwrap();
    out.len()
}

library_benchmark_group!(
    name = ab;
    benchmarks = len_intr, len_fear, encode_intr, encode_fear, decode_intr, decode_fear
);

main!(library_benchmark_groups = ab);
