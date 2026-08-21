# base94-simd

SIMD-accelerated base94 codec: encodes arbitrary binary data into the
printable ASCII range `0x20..=0x7E` (94 symbols). Each byte maps to `0x20 + b`; values `>= 93` escape into two chars
(`0x7D`/`0x7E` leader + follower), so every emitted char stays printable.

## Usage

```rust
let mut encoded = Vec::new();
base94_simd::encode_into(&mut encoded, b"hello");
assert!(encoded.iter().all(|&c| (0x20..=0x7e).contains(&c)));

let mut decoded = Vec::new();
base94_simd::decode_into(&mut decoded, &encoded).unwrap();
assert_eq!(decoded, b"hello");
```

Also provided: `encoded_len` (exact output size), `encode_into_with_len`
(skip the length scan), and `decimal_encode`/`decimal_decode` (base94
digits for integers).

## Backends

Picked at compile time / runtime, best first:

| target  | backend                                                              |
| ------- | -------------------------------------------------------------------- |
| x86_64  | AVX-512 F/BW/VBMI/VBMI2 + BMI2 kernels over 64-byte blocks (runtime-detected), SSSE3 kernels over 16-byte blocks chained for tails / older CPUs |
| aarch64 | baseline NEON kernels over 16-byte blocks                            |
| other   | portable scalar path (the reference definition)                      |

## Features

| feature   | effect                                                        |
| --------- | ------------------------------------------------------------- |
| `kf`      | Adds `encode_into_kf`/`decode_into_kf`/`encoded_len_kf`/`encode_into_with_len_kf`: key mixing (`(b - kf) mod 256` before encoding). |
| `hotpath` | Instruments the encode/decode entry points with [hotpath](https://crates.io/crates/hotpath). |

The `kf` mixing is compiled in via const generics.

## Performance

Measured on AMD 7945HX (single core, criterion, random binary input;
decode throughput counts encoded-text bytes):

| size   | encode     | decode     |
| ------ | ---------- | ---------- |
| 64 B   | 4.9 GiB/s  | 4.0 GiB/s  |
| 1 KiB  | 13.0 GiB/s | 10.9 GiB/s |
| 16 KiB | 12.8 GiB/s | 11.2 GiB/s |
| 1 MiB  | 13.7 GiB/s | 11.1 GiB/s |

Against the `base94` crate's original algorithm (BigUint base conversion,
O(n²)). Wire formats differ, but both turn binary into printable base94 text:

| size   | encode speedup | decode speedup |
| ------ | -------------- | -------------- |
| 64 B   | ~190x          | ~110x          |
| 1 KiB  | ~4300x         | ~1200x         |
| 16 KiB | ~69000x        | ~15000x        |

## Development

```sh
cargo test --features kf        # unit tests (both feature sets)
cargo bench                     # criterion suite incl. base94 comparison
cargo fuzz run roundtrip        # differential fuzzing (needs cargo-fuzz)
cargo run --release --features hotpath --example profile  # hotpath table
```
