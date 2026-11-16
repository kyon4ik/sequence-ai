//! Bounded retained Monte Carlo samples. Original outcomes are deliberately kept;
//! these are off-policy labels as the learner and opponent mixture change.
use super::*;
use std::{
    collections::VecDeque,
    io::{Read, Write},
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReplayConfig {
    pub capacity: usize,
    pub fraction: f64,
}
impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            capacity: 50_000,
            fraction: 0.25,
        }
    }
}
impl ReplayConfig {
    pub(super) fn validate(&self) -> Result<(), sequence_agent::AgentError> {
        if self.capacity == 0 || !(0.0..1.0).contains(&self.fraction) || self.fraction == 0. {
            return Err(
                "replay needs positive capacity and fraction strictly between zero and one".into(),
            );
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct ReplayBuffer {
    samples: VecDeque<(EncodedAfterstate, f32)>,
}
impl ReplayBuffer {
    pub fn draws(&self, count: usize, fraction: f64, seed: u64) -> BTreeMap<usize, usize> {
        if self.samples.is_empty() {
            return BTreeMap::new();
        }
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut slots: Vec<_> = (0..count).collect();
        slots.shuffle(&mut rng);
        slots
            .into_iter()
            .take((count as f64 * fraction).floor() as usize)
            .map(|slot| (slot, rng.random_range(0..self.samples.len())))
            .collect()
    }
    pub fn get(&self, index: usize) -> &(EncodedAfterstate, f32) {
        &self.samples[index]
    }
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    pub fn extend<'a>(
        &mut self,
        samples: impl Iterator<Item = (&'a EncodedAfterstate, f32)>,
        capacity: usize,
    ) {
        for (encoded, target) in samples {
            self.samples.push_back((encoded.clone(), target));
        }
        while self.samples.len() > capacity {
            self.samples.pop_front();
        }
    }
    pub fn save(&self, path: &Path) -> Result<(), sequence_agent::AgentError> {
        let mut bytes = Vec::with_capacity(16 + self.samples.len() * 1804);
        bytes.extend_from_slice(b"SEQRPL01");
        bytes.extend_from_slice(&(self.len() as u64).to_le_bytes());
        for (encoded, target) in &self.samples {
            for value in encoded
                .board
                .iter()
                .chain(&encoded.context)
                .chain(std::iter::once(target))
            {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        Ok(())
    }
    pub fn load(path: &Path, capacity: usize) -> Result<Self, sequence_agent::AgentError> {
        let mut file = std::fs::File::open(path)?;
        let mut header = [0u8; 16];
        file.read_exact(&mut header)?;
        if &header[..8] != b"SEQRPL01" {
            return Err("invalid replay buffer header".into());
        }
        let count = usize::try_from(u64::from_le_bytes(header[8..].try_into()?))?;
        if count > capacity
            || count
                .checked_mul(1804)
                .and_then(|n| n.checked_add(16))
                .map(|n| n as u64)
                != Some(file.metadata()?.len())
        {
            return Err("invalid replay buffer length".into());
        }
        let mut result = Self::default();
        let mut bytes = [0u8; 1804];
        for _ in 0..count {
            file.read_exact(&mut bytes)?;
            let values: Vec<_> = bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            if values.iter().any(|v| !v.is_finite()) || ![-1., 0., 1.].contains(&values[450]) {
                return Err("invalid replay sample".into());
            }
            result.samples.push_back((
                EncodedAfterstate {
                    board: values[..400].try_into()?,
                    context: values[400..450].try_into()?,
                },
                values[450],
            ));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn buffer_roundtrip_bounds_and_corruption() {
        let mut buffer = ReplayBuffer::default();
        let sample = EncodedAfterstate {
            board: [0.; 400],
            context: [0.; 50],
        };
        buffer.extend((0..12).map(|_| (&sample, 1.)), 8);
        assert_eq!(buffer.len(), 8);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("replay.bin");
        buffer.save(&path).unwrap();
        let loaded = ReplayBuffer::load(&path, 8).unwrap();
        assert_eq!(loaded.draws(100, 0.25, 42), buffer.draws(100, 0.25, 42));
        assert_eq!(loaded.draws(100, 0.25, 42).len(), 25);
        assert_eq!(loaded.get(0).0.board, buffer.get(0).0.board);
        assert_eq!(loaded.get(0).1, 1.);
        assert!(ReplayBuffer::load(&path, 7).is_err());
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[16..20].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        assert!(ReplayBuffer::load(&path, 8).is_err());
        assert!(
            ReplayConfig {
                fraction: f64::NAN,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}
