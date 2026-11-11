use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use sequence_agent::{AgentConfig, AgentDescriptor, AgentSpec};
use serde::{Serialize, de::DeserializeOwned};

use crate::validate_id;

pub struct Store {
    pub root: PathBuf,
}

impl Store {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn model_dir(&self, id: &str) -> Result<PathBuf> {
        validate_id(id)?;
        Ok(self.root.join("models").join(id))
    }

    #[cfg(feature = "burn")]
    pub fn reserve_model(&self, id: &str) -> Result<PathBuf> {
        if matches!(id, "random" | "greedy" | "heuristic") {
            anyhow::bail!("{id:?} is a reserved built-in agent ID");
        }
        let path = self.model_dir(id)?;
        fs::create_dir_all(self.root.join("models"))?;
        fs::create_dir(&path).with_context(|| {
            format!(
                "cannot create {}; choose a new agent ID if it already exists",
                path.display()
            )
        })?;
        Ok(path)
    }

    pub fn load_agent(&self, reference: &str, relative_to: &Path) -> Result<AgentSpec> {
        let config = match reference {
            "random" => Some(AgentConfig::Random),
            "greedy" | "heuristic" => Some(AgentConfig::greedy()),
            _ => None,
        };
        if let Some(config) = config {
            // Aliases share the same identity in rankings.
            let id = if reference == "heuristic" {
                "greedy"
            } else {
                reference
            };
            return Ok(AgentSpec {
                descriptor: AgentDescriptor::new(id, id),
                config,
            });
        }
        let supplied = relative_to.join(reference);
        let path = if supplied.is_file() {
            supplied
        } else {
            self.model_dir(reference)?.join("agent.json")
        };
        let mut spec: AgentSpec = read_json(&path).with_context(|| format!("unknown or unreadable agent {reference:?}; use random, greedy, a saved ID, or an agent.json path"))?;
        validate_id(&spec.descriptor.id)?;
        if let AgentConfig::MonteCarlo { checkpoint } = &mut spec.config {
            if checkpoint.is_relative() {
                *checkpoint = path.parent().unwrap_or(Path::new(".")).join(&*checkpoint);
            }
            *checkpoint = fs::canonicalize(&*checkpoint)
                .with_context(|| format!("cannot open checkpoint {}", checkpoint.display()))?;
        }
        Ok(spec)
    }

    pub fn replay_path(&self, id: &str) -> Result<PathBuf> {
        validate_id(id)?;
        Ok(self.root.join("replays").join(format!("{id}.json")))
    }

    pub fn load_replay(&self, reference: &str) -> Result<sequence_runner::MatchRecord> {
        let supplied = Path::new(reference);
        let path = if supplied.is_file() {
            supplied.to_path_buf()
        } else {
            self.replay_path(reference)?
        };
        read_json(&path)
    }

    pub fn save_replay(&self, id: &str, record: &sequence_runner::MatchRecord) -> Result<()> {
        fs::create_dir_all(self.root.join("replays"))?;
        write_json(&self.replay_path(id)?, record)
    }

    pub fn new_run(&self, kind: &str) -> Result<(String, PathBuf)> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let id = format!("{kind}-{nanos}-{}", std::process::id());
        let parent = self.root.join("runs");
        fs::create_dir_all(&parent)?;
        let path = parent.join(&id);
        fs::create_dir(&path)?;
        Ok((id, path))
    }
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_reader(file).with_context(|| format!("parsing {}", path.display()))
}

pub fn read_toml<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// Never replace an existing experiment artifact, even when its ID was reused.
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer
        .flush()
        .with_context(|| format!("flushing {}", path.display()))
}
