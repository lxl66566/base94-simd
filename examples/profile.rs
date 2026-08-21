//! Observable profiling harness for the codec hot path:
//!
//! ```sh
//! cargo run --release --features hotpath --example profile
//! ```

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

#[cfg_attr(feature = "hotpath", hotpath::main)]
fn main() {
    let data = sample(64 * 1024);
    let mut enc = Vec::new();
    let mut dec = Vec::new();
    for _ in 0..2000 {
        enc.clear();
        base94_simd::encode_into(&mut enc, &data, 0);
        dec.clear();
        base94_simd::decode_into(&mut dec, &enc, 0).unwrap();
    }
    assert_eq!(dec, data);
}
