// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! The `bloom-filter` file index, byte-compatible with Java's `BloomFilterFileIndex`.
//!
//! Serialized form: a 4-byte big-endian hash-function count followed by the raw bit set.
//! Values hash with Java's `FastHash`: byte sequences (binary/char/varchar) through XXH64
//! (seed 0), integer-family values through Thomas Wang's 64-bit mix. Membership testing uses
//! the standard double-hashing scheme over the 64-bit hash's two 32-bit halves, with Java's
//! exact overflow and sign behavior reproduced via wrapping arithmetic.

use crate::spec::Datum;
use crate::Error;

/// The index type identifier, as Java's `BloomFilterFileIndexFactory` registers it.
pub const BLOOM_FILTER_INDEX: &str = "bloom-filter";

/// A deserialized bloom-filter file index for one column.
pub struct BloomFilterIndex {
    num_hash_functions: i32,
    num_bits: i32,
    bits: Vec<u8>,
}

impl BloomFilterIndex {
    /// Deserializes Java's `[i32 BE numHashFunctions][bit set bytes]` layout.
    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        if bytes.len() <= 4 {
            return Err(Error::FileIndexFormatInvalid {
                message: format!("bloom filter index too short: {} bytes", bytes.len()),
            });
        }
        // Java deserializes with sign-propagating byte arithmetic; mirror i32::from_be_bytes.
        let num_hash_functions =
            i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let bits = bytes[4..].to_vec();
        Ok(BloomFilterIndex {
            num_hash_functions,
            num_bits: (bits.len() * 8) as i32,
            bits,
        })
    }

    /// Whether a value with the given 64-bit hash may be present (false = definitely absent).
    pub fn test_hash(&self, hash64: i64) -> bool {
        let hash1 = hash64 as i32;
        let hash2 = ((hash64 as u64) >> 32) as i32;
        for i in 1..=self.num_hash_functions {
            let mut combined = hash1.wrapping_add(i.wrapping_mul(hash2));
            if combined < 0 {
                combined = !combined;
            }
            let pos = (combined % self.num_bits) as usize;
            if self.bits[pos >> 3] & (1u8 << (pos & 7)) == 0 {
                return false;
            }
        }
        true
    }
}

/// A bloom-filter writer matching Java's sizing and serialization — for round-trip tests and
/// future write-side index support.
pub struct BloomFilterIndexWriter {
    num_hash_functions: i32,
    num_bits: i32,
    bits: Vec<u8>,
}

impl BloomFilterIndexWriter {
    pub fn new(items: i64, fpp: f64) -> Self {
        let nb = (-(items as f64) * fpp.ln() / (2f64.ln() * 2f64.ln())) as i32;
        let num_bits = nb + (8 - (nb % 8));
        let num_hash_functions =
            ((num_bits as f64 / items as f64 * 2f64.ln()).round() as i32).max(1);
        BloomFilterIndexWriter {
            num_hash_functions,
            num_bits,
            bits: vec![0u8; (num_bits / 8) as usize],
        }
    }

    pub fn add_hash(&mut self, hash64: i64) {
        let hash1 = hash64 as i32;
        let hash2 = ((hash64 as u64) >> 32) as i32;
        for i in 1..=self.num_hash_functions {
            let mut combined = hash1.wrapping_add(i.wrapping_mul(hash2));
            if combined < 0 {
                combined = !combined;
            }
            let pos = (combined % self.num_bits) as usize;
            self.bits[pos >> 3] |= 1u8 << (pos & 7);
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.bits.len());
        out.extend_from_slice(&self.num_hash_functions.to_be_bytes());
        out.extend_from_slice(&self.bits);
        out
    }
}

/// Java `FastHash.hash64(byte[])`: XXH64 with seed 0.
pub fn fast_hash_bytes(data: &[u8]) -> i64 {
    xxhash_rust::xxh64::xxh64(data, 0) as i64
}

/// Java `FastHash.getLongHash`: Thomas Wang's 64-bit integer mix, with Java's wrapping overflow
/// and arithmetic (sign-propagating) shifts.
pub fn fast_hash_long(key: i64) -> i64 {
    let mut key = (!key).wrapping_add(key.wrapping_shl(21));
    key ^= key >> 24;
    key = key.wrapping_add(key.wrapping_shl(3)).wrapping_add(key.wrapping_shl(8));
    key ^= key >> 14;
    key = key.wrapping_add(key.wrapping_shl(2)).wrapping_add(key.wrapping_shl(4));
    key ^= key >> 28;
    key.wrapping_add(key.wrapping_shl(31))
}

/// The `FastHash` of a predicate literal, or `None` when the type has no fast hash (the caller
/// must keep the file — never skip on an unhashable literal).
pub fn fast_hash_datum(datum: &Datum) -> Option<i64> {
    match datum {
        Datum::Bytes(bytes) => Some(fast_hash_bytes(bytes)),
        Datum::String(s) => Some(fast_hash_bytes(s.as_bytes())),
        Datum::TinyInt(v) => Some(fast_hash_long(*v as i64)),
        Datum::SmallInt(v) => Some(fast_hash_long(*v as i64)),
        Datum::Int(v) => Some(fast_hash_long(*v as i64)),
        Datum::Long(v) => Some(fast_hash_long(*v)),
        Datum::Date(v) => Some(fast_hash_long(*v as i64)),
        Datum::Time(v) => Some(fast_hash_long(*v as i64)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wang_hash_matches_java_vectors() {
        // Ground truth computed by running Java's FastHash.getLongHash verbatim.
        assert_eq!(fast_hash_long(0), 0);
        assert_eq!(fast_hash_long(1), 6614235796240398542);
        assert_eq!(fast_hash_long(-1), 6614246905173314819);
        assert_eq!(fast_hash_long(123456789), -1864789099685094664);
        assert_eq!(fast_hash_long(i64::MAX), -9102528624353286089);
        assert_eq!(fast_hash_long(i64::MIN), 4316648529147585864);
    }

    #[test]
    fn xx_hash_matches_java_vectors() {
        // Ground truth computed by running Java's LongHashFunction.xx().hashBytes verbatim
        // (zero-allocation-hashing 0.26 — standard XXH64, seed 0).
        assert_eq!(fast_hash_bytes(&[]), -1205034819632174695);
        assert_eq!(fast_hash_bytes(b"abc"), 4952883123889572249);
        assert_eq!(fast_hash_bytes(&[1, 2, 3, 4, 5, 6, 7, 8]), -9129847667296014828);
    }

    #[test]
    fn bloom_round_trips_and_rejects_absent() {
        let mut writer = BloomFilterIndexWriter::new(1000, 0.01);
        for i in 0..1000i64 {
            writer.add_hash(fast_hash_long(i));
        }
        let index = BloomFilterIndex::from_bytes(&writer.serialize()).unwrap();
        for i in 0..1000i64 {
            assert!(index.test_hash(fast_hash_long(i)), "present key {i} must remain");
        }
        let misses = (10_000..20_000i64)
            .filter(|&i| index.test_hash(fast_hash_long(i)))
            .count();
        assert!(misses < 300, "false-positive rate out of range: {misses}/10000");
    }
}
