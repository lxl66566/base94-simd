//! Throughput benches: base94-simd vs the original base94 crate algorithm.
//!
//! The `base94` crate itself is intentionally not a dependency; its 0.3.0
//! algorithm (BigUint base conversion, MIT) is copied below verbatim in
//! structure, so the comparison measures the algorithm, not a rewrite.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::ToPrimitive;

// ---- Copied from base94 0.3.0 (Antosser/base94-rs, MIT) ----

const CHARACTERS: &[u8; 94] = b"!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";

fn base94_encode(data: &[u8], base: u8) -> String {
    assert!(base <= 94, "Base must be less than or equal to 94");
    assert!(base >= 2, "Base must be greater than or equal to 2");

    let mut num = BigUint::from_bytes_le(data);
    let mut out = String::new();

    while num > BigUint::from(0u8) {
        let (div, rem) = num.div_rem(&BigUint::from(base));
        num = div;
        out.push(CHARACTERS[rem.to_usize().unwrap()] as char);
    }

    out
}

fn base94_decode(encoded: &str, base: u8) -> Vec<u8> {
    let mut num = BigUint::from(0u8);
    let mut power = BigUint::from(1u8);

    for c in encoded.chars() {
        let index = CHARACTERS.iter().position(|&x| x == c as u8).unwrap();
        num += BigUint::from(index) * &power;
        power *= BigUint::from(base);
    }

    num.to_bytes_le()
}

// ---- bench harness ----

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

fn bench_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("encode");
    for size in [64usize, 1024, 16 * 1024, 64 * 1024, 1024 * 1024] {
        let data = sample(size);
        g.throughput(Throughput::Bytes(size as u64));
        g.bench_with_input(BenchmarkId::new("base94-simd", size), &data, |b, data| {
            let mut out = Vec::new();
            b.iter(|| {
                out.clear();
                base94_simd::encode_into(black_box(&mut out), black_box(data));
            });
        });
        if size <= 16 * 1024 {
            g.bench_with_input(
                BenchmarkId::new("base94-biguint", size),
                &data,
                |b, data| {
                    b.iter(|| base94_encode(black_box(data), 94));
                },
            );
        }
    }
    g.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("decode");
    for size in [64usize, 1024, 16 * 1024, 64 * 1024, 1024 * 1024] {
        let data = sample(size);
        let mut encoded = Vec::new();
        base94_simd::encode_into(&mut encoded, &data);
        g.throughput(Throughput::Bytes(encoded.len() as u64));
        g.bench_with_input(BenchmarkId::new("base94-simd", size), &encoded, |b, enc| {
            let mut out = Vec::new();
            b.iter(|| {
                out.clear();
                base94_simd::decode_into(black_box(&mut out), black_box(enc)).unwrap();
            });
        });
        if size <= 16 * 1024 {
            let enc94 = base94_encode(&data, 94);
            g.bench_with_input(
                BenchmarkId::new("base94-biguint", size),
                &enc94,
                |b, enc| {
                    b.iter(|| base94_decode(black_box(enc), 94));
                },
            );
        }
    }
    g.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
