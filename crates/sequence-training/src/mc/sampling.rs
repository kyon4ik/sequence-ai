use super::{OpponentKind, OpponentMix};
use rand::{RngExt, seq::SliceRandom};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStrategy {
    /// All fresh samples once by default; uniform sampling with replacement
    /// when updates_per_iteration supplies a fixed budget.
    #[default]
    Natural,
    /// Sample categories according to the requested opponent game weights.
    /// Samples within each category are drawn uniformly with replacement.
    OpponentMix,
}

pub(super) fn select(
    sources: &[OpponentKind],
    mix: &OpponentMix,
    strategy: SamplingStrategy,
    count: Option<usize>,
    iteration: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<usize> {
    if strategy == SamplingStrategy::Natural {
        return match count {
            None => (0..sources.len()).collect(),
            Some(count) => (0..count)
                .map(|_| rng.random_range(0..sources.len()))
                .collect(),
        };
    }
    let mut buckets: BTreeMap<_, Vec<usize>> = BTreeMap::new();
    for (index, &source) in sources.iter().enumerate() {
        buckets.entry(source).or_default().push(index);
    }
    let mut available = mix.clone();
    // Very small rollout budgets can round a category down to zero games. Only
    // categories with fresh samples can participate in this iteration's updates.
    let weights = [
        &mut available.self_play,
        &mut available.best_checkpoint,
        &mut available.previous_checkpoint,
        &mut available.random_checkpoint,
        &mut available.greedy,
    ];
    for (source, weight) in OpponentKind::ALL.into_iter().zip(weights) {
        if !buckets.contains_key(&source) {
            *weight = 0.;
        }
    }
    let count = count.unwrap_or(sources.len());
    let mut indices = Vec::with_capacity(count);
    for (source, count) in OpponentKind::ALL
        .into_iter()
        .zip(available.counts(count, iteration))
    {
        if count == 0 {
            continue;
        }
        let bucket = &buckets[&source];
        indices.extend((0..count).map(|_| bucket[rng.random_range(0..bucket.len())]));
    }
    indices.shuffle(rng);
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn balanced_sampling_matches_weights_despite_unequal_source_sizes() {
        let mix = OpponentMix {
            self_play: 40.,
            best_checkpoint: 20.,
            previous_checkpoint: 15.,
            random_checkpoint: 15.,
            greedy: 10.,
        };
        let sources: Vec<_> = OpponentKind::ALL
            .into_iter()
            .zip([600, 150, 100, 100, 50])
            .flat_map(|(source, n)| std::iter::repeat_n(source, n))
            .collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(71);
        let mut restored = rng.clone();
        let indices = select(
            &sources,
            &mix,
            SamplingStrategy::OpponentMix,
            Some(1000),
            0,
            &mut rng,
        );
        assert_eq!(
            indices,
            select(
                &sources,
                &mix,
                SamplingStrategy::OpponentMix,
                Some(1000),
                0,
                &mut restored
            )
        );
        let counts =
            OpponentKind::ALL.map(|kind| indices.iter().filter(|&&i| sources[i] == kind).count());
        assert_eq!(counts, [400, 200, 150, 150, 100]);
        assert_eq!(
            select(&sources, &mix, SamplingStrategy::Natural, None, 0, &mut rng),
            (0..sources.len()).collect::<Vec<_>>()
        );
        assert_eq!(
            select(
                &sources,
                &mix,
                SamplingStrategy::Natural,
                Some(123),
                0,
                &mut rng
            )
            .len(),
            123
        );
    }

    #[test]
    fn absent_categories_are_renormalized() {
        let mix = OpponentMix {
            self_play: 40.,
            greedy: 10.,
            ..Default::default()
        };
        let sources = [OpponentKind::Greedy; 5];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(72);
        let indices = select(
            &sources,
            &mix,
            SamplingStrategy::OpponentMix,
            Some(100),
            0,
            &mut rng,
        );
        assert_eq!(indices.len(), 100);
        assert!(indices.iter().all(|&i| sources[i] == OpponentKind::Greedy));
    }
}
