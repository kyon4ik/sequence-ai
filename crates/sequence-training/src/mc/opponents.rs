use burn::prelude::*;
use rand::{RngExt, SeedableRng, seq::SliceRandom};
use rand_xoshiro::Xoshiro256PlusPlus;
use sequence_agent::{AgentError, mc::ValueNet};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    collections::BTreeSet,
    path::{Path, PathBuf},
};

/// Relative proportions of games, rather than gradient samples. An omitted
/// table defaults to self-play; omitted entries in a supplied table are zero.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpponentMix {
    #[serde(default)]
    pub self_play: f64,
    #[serde(default)]
    pub best_checkpoint: f64,
    #[serde(default)]
    pub previous_checkpoint: f64,
    #[serde(default)]
    pub random_checkpoint: f64,
    #[serde(default)]
    pub greedy: f64,
}

impl Default for OpponentMix {
    fn default() -> Self {
        Self {
            self_play: 100.,
            best_checkpoint: 0.,
            previous_checkpoint: 0.,
            random_checkpoint: 0.,
            greedy: 0.,
        }
    }
}

impl OpponentMix {
    fn weights(&self) -> [f64; 5] {
        [
            self.self_play,
            self.best_checkpoint,
            self.previous_checkpoint,
            self.random_checkpoint,
            self.greedy,
        ]
    }

    pub(super) fn validate(&self) -> Result<(), AgentError> {
        let weights = self.weights();
        let total: f64 = weights.iter().sum();
        if weights.iter().any(|w| !w.is_finite() || *w < 0.) || !total.is_finite() || total <= 0. {
            return Err(
                "opponent weights must be finite, nonnegative and have a positive sum".into(),
            );
        }
        Ok(())
    }

    pub(super) fn uses_history(&self) -> bool {
        self.best_checkpoint > 0. || self.previous_checkpoint > 0. || self.random_checkpoint > 0.
    }

    pub(super) fn is_self_play(&self) -> bool {
        self.self_play > 0. && !self.uses_history() && self.greedy == 0.
    }

    /// Largest-remainder allocation; rotate equal remainders between iterations.
    pub(super) fn counts(&self, games: usize, iteration: usize) -> [usize; 5] {
        let weights = self.weights();
        let total: f64 = weights.iter().sum();
        let quotas = weights.map(|weight| weight / total * games as f64);
        let mut counts = quotas.map(|quota| quota.floor() as usize);
        let mut order: Vec<_> = (0..5).filter(|&i| weights[i] > 0.).collect();
        order.sort_by(|&a, &b| {
            (quotas[b] - counts[b] as f64)
                .total_cmp(&(quotas[a] - counts[a] as f64))
                .then_with(|| ((a + 5 - iteration % 5) % 5).cmp(&((b + 5 - iteration % 5) % 5)))
        });
        for &index in order.iter().take(games - counts.iter().sum::<usize>()) {
            counts[index] += 1;
        }
        counts
    }

    pub(super) fn schedule(
        &self,
        games: usize,
        completed: usize,
        state: &mut OpponentState,
    ) -> Vec<OpponentPlan> {
        // Preserve legacy rollout/shuffle RNG consumption and game order.
        if self.is_self_play() {
            return (0..games)
                .map(|index| OpponentPlan {
                    source: OpponentKind::SelfPlay,
                    policy: OpponentPolicy::SelfPlay,
                    learner_seat: index % 2,
                })
                .collect();
        }
        let historical: Vec<_> = state
            .checkpoints
            .iter()
            .copied()
            .filter(|&i| i < completed)
            .collect();
        let mut plans = Vec::with_capacity(games);
        for (kind_index, (source, count)) in OpponentKind::ALL
            .into_iter()
            .zip(self.counts(games, completed))
            .enumerate()
        {
            for index in 0..count {
                let policy = match source {
                    OpponentKind::SelfPlay => OpponentPolicy::SelfPlay,
                    OpponentKind::Greedy => OpponentPolicy::Greedy,
                    OpponentKind::BestCheckpoint => state
                        .best_iteration
                        .map(OpponentPolicy::Checkpoint)
                        .unwrap_or(OpponentPolicy::SelfPlay),
                    OpponentKind::PreviousCheckpoint => completed
                        .checked_sub(1)
                        .filter(|i| historical.contains(i))
                        .map(OpponentPolicy::Checkpoint)
                        .unwrap_or(OpponentPolicy::SelfPlay),
                    OpponentKind::RandomCheckpoint => {
                        if historical.is_empty() {
                            OpponentPolicy::SelfPlay
                        } else {
                            OpponentPolicy::Checkpoint(
                                historical[state.rng.random_range(0..historical.len())],
                            )
                        }
                    }
                };
                plans.push(OpponentPlan {
                    source,
                    policy,
                    learner_seat: (index + completed + kind_index) % 2,
                });
            }
        }
        plans.shuffle(&mut state.rng);
        plans
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpponentKind {
    #[default]
    SelfPlay,
    BestCheckpoint,
    PreviousCheckpoint,
    RandomCheckpoint,
    Greedy,
}

impl OpponentKind {
    pub(super) const ALL: [Self; 5] = [
        Self::SelfPlay,
        Self::BestCheckpoint,
        Self::PreviousCheckpoint,
        Self::RandomCheckpoint,
        Self::Greedy,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum OpponentPolicy {
    SelfPlay,
    Checkpoint(usize),
    Greedy,
}

impl OpponentPolicy {
    pub fn id(self) -> String {
        match self {
            Self::SelfPlay => "self-play".into(),
            Self::Greedy => "greedy".into(),
            Self::Checkpoint(iteration) => format!("iteration-{iteration:06}"),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct OpponentPlan {
    pub source: OpponentKind,
    pub policy: OpponentPolicy,
    pub learner_seat: usize,
}

impl OpponentPlan {
    pub fn learns_from(&self, actor: usize) -> bool {
        self.policy == OpponentPolicy::SelfPlay || actor == self.learner_seat
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OpponentMetrics {
    pub source: OpponentKind,
    pub opponent: String,
    pub games: usize,
    pub samples: usize,
    /// Samples used for gradient updates, including repeated selections.
    #[serde(default)]
    pub optimization_samples: usize,
    pub learner_seats: [usize; 2],
    /// Wins, losses, draws from the designated learner seat's perspective.
    pub outcomes: [usize; 3],
    pub fallback_games: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpponentEvaluation {
    pub opponent: String,
    pub games: usize,
    /// Wins, losses, draws from the learner's perspective.
    pub outcomes: [usize; 3],
    pub score: f64,
}

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct OpponentState {
    pub rng: Xoshiro256PlusPlus,
    pub checkpoints: Vec<usize>,
    pub best_iteration: Option<usize>,
}

impl OpponentState {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed ^ 0x0FF0_4E47),
            checkpoints: Vec::new(),
            best_iteration: None,
        }
    }
}

/// Immutable weight-only archive on disk. Published snapshots include hard links
/// (copies across filesystems), so each snapshot can be moved and resumed alone.
pub(super) struct CheckpointPool {
    directory: RefCell<tempfile::TempDir>,
    iterations: BTreeSet<usize>,
}

impl CheckpointPool {
    pub fn new() -> Result<Self, AgentError> {
        Ok(Self {
            directory: RefCell::new(tempfile::tempdir()?),
            iterations: BTreeSet::new(),
        })
    }

    pub fn path(&self, iteration: usize) -> PathBuf {
        self.directory
            .borrow()
            .path()
            .join(format!("iteration-{iteration:06}.bpk"))
    }

    pub fn insert(&mut self, iteration: usize, net: &ValueNet) -> Result<(), AgentError> {
        if self.iterations.contains(&iteration) {
            return Err("duplicate opponent checkpoint".into());
        }
        net.clone().valid().save_file(self.path(iteration))?;
        self.iterations.insert(iteration);
        Ok(())
    }

    pub fn restore(&mut self, snapshot: &Path, iterations: &[usize]) -> Result<(), AgentError> {
        // Keep the owned archive beside the source when writable, so restoring
        // a large history does not copy it into a potentially memory-backed /tmp.
        if let Ok(directory) = tempfile::tempdir_in(snapshot.parent().unwrap_or(Path::new("."))) {
            self.directory = RefCell::new(directory);
        }
        for &iteration in iterations {
            if !self.iterations.insert(iteration) {
                return Err("duplicate opponent checkpoint in manifest".into());
            }
            link_or_copy(
                &snapshot
                    .join("opponents")
                    .join(format!("iteration-{iteration:06}.bpk")),
                &self.path(iteration),
            )?;
        }
        Ok(())
    }

    pub fn save(&self, snapshot: &Path) -> Result<(), AgentError> {
        let directory = snapshot.join("opponents");
        std::fs::create_dir(&directory)?;
        let first = self
            .iterations
            .first()
            .ok_or("empty opponent checkpoint pool")?;
        let first_destination = directory.join(format!("iteration-{first:06}.bpk"));
        let first_linked = std::fs::hard_link(self.path(*first), &first_destination).is_ok();
        if !first_linked {
            // The callback supplies the storage filesystem for the first time.
            // Relocate our owned archive there once, then share immutable files
            // with every snapshot rather than copying the entire history again.
            let archive = tempfile::tempdir_in(snapshot.parent().unwrap_or(Path::new(".")))?;
            for &iteration in &self.iterations {
                let source = self.path(iteration);
                link_or_copy(&source, &archive.path().join(source.file_name().unwrap()))?;
            }
            *self.directory.borrow_mut() = archive;
        }
        for &iteration in &self.iterations {
            let source = self.path(iteration);
            let destination = directory.join(source.file_name().unwrap());
            if iteration != *first || !first_linked {
                link_or_copy(&source, &destination)?;
            }
            std::fs::File::open(destination)?.sync_all()?;
        }
        std::fs::File::open(directory)?.sync_all()?;
        Ok(())
    }
}

fn link_or_copy(source: &Path, destination: &Path) -> Result<(), AgentError> {
    if std::fs::hard_link(source, destination).is_err() {
        std::fs::copy(source, destination)
            .map_err(|e| format!("copying opponent checkpoint {}: {e}", source.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn mixed() -> OpponentMix {
        OpponentMix {
            self_play: 40.,
            best_checkpoint: 20.,
            previous_checkpoint: 15.,
            random_checkpoint: 15.,
            greedy: 10.,
        }
    }

    #[test]
    fn allocation_seats_history_and_saved_rng() {
        let mix = mixed();
        assert_eq!(mix.counts(100, 0), [40, 20, 15, 15, 10]);
        assert_eq!(mix.counts(128, 0), [51, 26, 19, 19, 13]);
        let mut state = OpponentState::new(43);
        state.checkpoints = (0..=5).collect();
        state.best_iteration = Some(3);
        let saved = serde_json::to_vec(&state).unwrap();
        let mut restored = serde_json::from_slice(&saved).unwrap();
        let plans = mix.schedule(100, 5, &mut state);
        let resumed = mix.schedule(100, 5, &mut restored);
        assert_eq!(
            plans
                .iter()
                .map(|p| (p.source, p.policy, p.learner_seat))
                .collect::<Vec<_>>(),
            resumed
                .iter()
                .map(|p| (p.source, p.policy, p.learner_seat))
                .collect::<Vec<_>>()
        );
        for kind in OpponentKind::ALL {
            let group: Vec<_> = plans.iter().filter(|p| p.source == kind).collect();
            let red = group.iter().filter(|p| p.learner_seat == 0).count();
            assert!(red.abs_diff(group.len() - red) <= 1);
            for plan in group {
                match kind {
                    OpponentKind::BestCheckpoint => {
                        assert_eq!(plan.policy, OpponentPolicy::Checkpoint(3))
                    }
                    OpponentKind::PreviousCheckpoint => {
                        assert_eq!(plan.policy, OpponentPolicy::Checkpoint(4))
                    }
                    OpponentKind::RandomCheckpoint => {
                        assert!(matches!(plan.policy, OpponentPolicy::Checkpoint(i) if i < 5))
                    }
                    _ => {}
                }
            }
        }
        let early = mix.schedule(100, 0, &mut OpponentState::new(43));
        assert_eq!(
            early
                .iter()
                .filter(|p| p.policy == OpponentPolicy::SelfPlay)
                .count(),
            90
        );
    }

    #[test]
    fn weights_are_validated_and_partial_tables_have_zero_defaults() {
        let mix: OpponentMix = serde_json::from_str(r#"{"greedy": 10}"#).unwrap();
        assert_eq!(mix.self_play, 0.);
        assert_eq!(mix.counts(7, 0), [0, 0, 0, 0, 7]);
        for invalid in [0., -1., f64::NAN, f64::INFINITY] {
            let mut mix = mix.clone();
            mix.greedy = invalid;
            assert!(mix.validate().is_err());
        }
        assert!(serde_json::from_str::<OpponentMix>(r#"{"greddy": 10}"#).is_err());
    }

    #[test]
    fn snapshots_share_weights_and_survive_the_runtime_archive() {
        // Use the workspace filesystem: /tmp may be a separate memory-backed
        // mount, which must trigger archive relocation rather than repeated copies.
        let output = tempfile::tempdir_in(std::env::current_dir().unwrap()).unwrap();
        let first = output.path().join("first");
        let second = output.path().join("second");
        std::fs::create_dir(&first).unwrap();
        std::fs::create_dir(&second).unwrap();
        let mut pool = CheckpointPool::new().unwrap();
        let net = ValueNet::new(&Device::flex());
        pool.insert(0, &net).unwrap();
        pool.save(&first).unwrap();
        pool.insert(1, &net).unwrap();
        pool.save(&second).unwrap();
        let name = "opponents/iteration-000000.bpk";
        assert_eq!(
            std::fs::read(first.join(name)).unwrap(),
            std::fs::read(second.join(name)).unwrap()
        );
        #[cfg(unix)]
        if std::fs::hard_link(first.join(name), output.path().join("link-probe")).is_ok() {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(
                std::fs::metadata(first.join(name)).unwrap().ino(),
                std::fs::metadata(second.join(name)).unwrap().ino()
            );
        }
        std::fs::remove_dir_all(&first).unwrap();
        assert!(pool.path(0).is_file());
        drop(pool);
        assert!(second.join(name).is_file());
        assert!(second.join("opponents/iteration-000001.bpk").is_file());
    }
}
