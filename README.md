# base94-simd

SIMD-accelerated base94 codec: encodes arbitrary binary data into the
printable ASCII range `0x20..=0x7E` (94 symbols), with an optional key-mixing
step (`kf`) and a base94 digit encoding for integers.

Bytes are key-mixed (`(b - kf) mod 256`); mixed values `>= 93` escape into two
chars (`0x7D`/`0x7E` leader + follower), so output is at most twice the input
size. With `kf = 0` the mixing is the identity.

## Backends

- x86_64 (runtime-detected, best first):
  - AVX-512 F/BW/VBMI/VBMI2 + BMI2 kernels over 64-byte blocks: masks live in
    scalar `u64`/`__mmask64`, escape reconstruction runs in u8 lanes, and
    output compaction is a single `vpcompressb` per 64-byte block/half.
  - SSSE3 kernels over 16-byte blocks (pshufb compaction LUT), chained for
    the 16..63-byte tail behind the AVX-512 kernels.
- aarch64: baseline NEON kernels over 16-byte blocks (`vqtbl1` shares the
  compaction LUT with the pshufb backend; no runtime detection needed).
- other targets: portable scalar path (the reference definition).

All backends are bit-exact, pinned by fuzzing against the scalar reference.

## Performance

Measured on an AMD Zen 5 VPS (single core, criterion, random binary input,
`kf = 0`; decode throughput counts encoded-text bytes):

| size  | encode       | decode       |
| ----- | ------------ | ------------ |
| 64 B  | 4.8 GiB/s    | 3.9 GiB/s    |
| 1 KiB | 12.8 GiB/s   | 10.8 GiB/s   |
| 16 KiB| 13.7 GiB/s   | 11.0 GiB/s   |
| 1 MiB | 13.4 GiB/s   | 10.9 GiB/s   |

That is ~4.4x (encode) / ~3.2x (decode) over the previous SSSE3-only
kernels (~3.1–3.7 GiB/s) on the same machine.

Comparison with the `base94` crate (v0.3.0): its original algorithm
(BigUint base conversion, O(n²)) is copied into `benches/codec.rs` purely
for benchmarking — the crate itself is **not** a dependency. Note the wire
formats differ; both turn binary data into printable base94 text, so the
throughput ratio is the meaningful axis:

| size  | encode speedup | decode speedup |
| ----- | -------------- | -------------- |
| 64 B  | ~190x          | ~110x          |
| 1 KiB | ~4300x         | ~1200x         |
| 16 KiB| ~69000x        | ~15000x        |

(the BigUint algorithm is quadratic, so the ratio grows with size; its
absolute throughput at 16 KiB is ~207 KiB/s encode / ~785 KiB/s decode.)

Profiling: `cargo run --release --features hotpath --example profile`
prints a [hotpath](https://crates.io/crates/hotpath) table of the measured
entry points; `cargo bench` runs the criterion suite.

```rust
let mut encoded = Vec::new();
base94_simd::encode_into(&mut encoded, b"hello", 0);
assert!(encoded.iter().all(|&c| (0x20..=0x7e).contains(&c)));
let mut decoded = Vec::new();
base94_simd::decode_into(&mut decoded, &encoded, 0).unwrap();
assert_eq!(decoded, b"hello");
```
