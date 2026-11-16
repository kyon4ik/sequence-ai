use crate::{Agent, AgentDescriptor, AgentError, AgentFactory, GreedyAgent, RandomAgent};
use sequence_core::Simple;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Portable policy parameters. Callers resolve checkpoint paths; artifact
/// management belongs to the application.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentConfig {
    Random,
    Greedy { weights: [i32; 6] },
    MonteCarlo { checkpoint: PathBuf },
}

impl AgentConfig {
    pub fn greedy() -> Self {
        Self::Greedy {
            weights: GreedyAgent::DEFAULT_WEIGHTS,
        }
    }

    pub fn create(
        &self,
        seed: u64,
        device: InferenceDevice,
    ) -> Result<Box<dyn Agent<Simple>>, BuildError> {
        match self {
            Self::Random => Ok(Box::new(RandomAgent::from_seed(seed))),
            Self::Greedy { weights } => Ok(Box::new(GreedyAgent::with_weights(*weights))),
            Self::MonteCarlo { checkpoint } => {
                #[cfg(feature = "burn")]
                {
                    use burn::prelude::*;
                    std::fs::File::open(checkpoint).map_err(|e| BuildError::Checkpoint {
                        path: checkpoint.clone(),
                        message: e.to_string(),
                    })?;
                    let device = device.burn_device();
                    let model = crate::mc::ValueNet::new(&device)
                        .try_load_file(checkpoint)
                        .map_err(|e| BuildError::Checkpoint {
                            path: checkpoint.clone(),
                            message: e.to_string(),
                        })?;
                    Ok(Box::new(crate::mc::McAgent::new(model, device, seed, 0.0)))
                }
                #[cfg(not(feature = "burn"))]
                {
                    let _ = (checkpoint, device);
                    Err(BuildError::BurnDisabled)
                }
            }
        }
    }
}

/// Runtime hardware choice, separate from saved policy parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InferenceDevice {
    #[default]
    Cpu,
    Cuda(usize),
}

#[cfg(feature = "burn")]
impl InferenceDevice {
    pub fn burn_device(self) -> burn::prelude::Device {
        match self {
            Self::Cpu => burn::prelude::Device::flex(),
            Self::Cuda(index) => burn::prelude::Device::cuda(index),
        }
    }
}

/// Serialize beside a model checkpoint. Heuristics need no separate artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentSpec {
    pub descriptor: AgentDescriptor,
    pub config: AgentConfig,
}

impl AgentFactory<Simple> for AgentSpec {
    fn descriptor(&self) -> AgentDescriptor {
        self.descriptor.clone()
    }
    fn create(&self, seed: u64) -> Result<Box<dyn Agent<Simple>>, AgentError> {
        Ok(self.config.create(seed, InferenceDevice::Cpu)?)
    }
}

impl AgentSpec {
    pub fn on_device(&self, device: InferenceDevice) -> impl AgentFactory<Simple> + '_ {
        struct OnDevice<'a> {
            spec: &'a AgentSpec,
            device: InferenceDevice,
        }
        impl AgentFactory<Simple> for OnDevice<'_> {
            fn descriptor(&self) -> AgentDescriptor {
                self.spec.descriptor.clone()
            }
            fn create(&self, seed: u64) -> Result<Box<dyn Agent<Simple>>, AgentError> {
                Ok(self.spec.config.create(seed, self.device)?)
            }
        }
        OnDevice { spec: self, device }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("Monte Carlo agents require the sequence-agent/burn feature")]
    BurnDisabled,
    #[error("cannot load checkpoint {path}: {message}")]
    Checkpoint { path: PathBuf, message: String },
}
