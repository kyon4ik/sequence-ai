//! Complete training checkpoint publication and loading.
use super::{
    TrainingConfig, TrainingReport,
    opponents::{CheckpointPool, OpponentState},
    replay_buffer::ReplayBuffer,
};
use burn::{optim::ModuleOptimizer, prelude::*};
use rand_xoshiro::Xoshiro256PlusPlus;
use sequence_agent::mc::ValueNet;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub(super) const CHECKPOINT_VERSION: u32 = 3;

#[derive(Serialize, Deserialize)]
pub(super) struct ResumeMetadata {
    pub(super) version: u32,
    pub(super) config: TrainingConfig,
    pub(super) rng: Xoshiro256PlusPlus,
    pub(super) report: TrainingReport,
    pub(super) best_score: Option<f64>,
    pub(super) opponents: OpponentState,
}

pub struct TrainingCheckpoint<'a> {
    pub(super) model: &'a ValueNet,
    pub(super) optimizer: &'a ModuleOptimizer,
    pub(super) metadata: &'a ResumeMetadata,
    pub(super) best_model: Option<&'a ValueNet>,
    pub(super) pool: Option<&'a CheckpointPool>,
    pub(super) replay: Option<&'a ReplayBuffer>,
}

impl TrainingCheckpoint<'_> {
    pub fn report(&self) -> &TrainingReport {
        &self.metadata.report
    }

    /// Publish a complete checkpoint directory; incomplete writes stay invisible.
    pub fn save(&self, destination: &Path) -> Result<(), sequence_agent::AgentError> {
        if destination.exists() {
            return Err("checkpoint already exists".into());
        }
        let parent = destination.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let staging = tempfile::tempdir_in(parent)?;
        self.model
            .clone()
            .valid()
            .save_file(staging.path().join("model.bpk"))?;
        self.optimizer.save(staging.path().join("optimizer.bpk"))?;
        if let Some(best) = self.best_model {
            best.clone()
                .valid()
                .save_file(staging.path().join("best.bpk"))?;
        }
        if let Some(pool) = self.pool {
            pool.save(staging.path())?;
        }
        if let Some(replay) = self.replay {
            replay.save(&staging.path().join("replay.bin"))?;
        }
        std::fs::write(
            staging.path().join("state.json"),
            serde_json::to_vec_pretty(self.metadata)?,
        )?;
        for entry in std::fs::read_dir(staging.path())? {
            std::fs::File::open(entry?.path())?.sync_all()?;
        }
        std::fs::rename(staging.path(), destination)?;
        Ok(())
    }
}

impl ResumeMetadata {
    pub(super) fn load(path: &Path) -> Result<Self, sequence_agent::AgentError> {
        let metadata: Self =
            serde_json::from_reader(std::fs::File::open(path.join("state.json"))?)?;
        if metadata.version != CHECKPOINT_VERSION {
            return Err("unsupported training checkpoint version".into());
        }
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn loading_requires_current_version_and_complete_scheduler() {
        let metadata = ResumeMetadata {
            version: CHECKPOINT_VERSION,
            config: TrainingConfig::default(),
            rng: Xoshiro256PlusPlus::seed_from_u64(42),
            report: TrainingReport::default(),
            best_score: None,
            opponents: OpponentState::new(42),
        };
        let root = tempfile::tempdir().unwrap();
        let original = serde_json::to_value(metadata).unwrap();
        let save = |state: &serde_json::Value| {
            std::fs::write(
                root.path().join("state.json"),
                serde_json::to_vec(state).unwrap(),
            )
            .unwrap();
        };
        save(&original);
        ResumeMetadata::load(root.path()).unwrap();
        for version in [1, 2, CHECKPOINT_VERSION + 1] {
            let mut state = original.clone();
            state["version"] = version.into();
            save(&state);
            assert!(ResumeMetadata::load(root.path()).is_err());
        }
        let mut state = original.clone();
        state.as_object_mut().unwrap().remove("opponents");
        save(&state);
        assert!(ResumeMetadata::load(root.path()).is_err());
        let mut state = original;
        state["opponents"]
            .as_object_mut()
            .unwrap()
            .remove("qualities");
        save(&state);
        assert!(ResumeMetadata::load(root.path()).is_err());
    }
}
