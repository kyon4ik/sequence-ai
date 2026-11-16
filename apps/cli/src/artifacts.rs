use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
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
        if matches!(id, "random" | "greedy") {
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
            "greedy" => Some(AgentConfig::greedy()),
            _ => None,
        };
        if let Some(config) = config {
            return Ok(AgentSpec {
                descriptor: AgentDescriptor::new(reference, reference),
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
        let (folder, game) = id
            .split_once('/')
            .context("expected a folder/game replay reference")?;
        validate_id(folder)?;
        validate_id(game)?;
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
        let path = self.replay_path(id)?;
        fs::create_dir_all(path.parent().unwrap())?;
        write_json(&path, record)
    }

    pub fn replay_folders(&self) -> Result<Vec<String>> {
        let root = self.root.join("replays");
        let mut folders = Vec::new();
        if !root.exists() {
            return Ok(folders);
        }
        for entry in fs::read_dir(&root)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if entry.file_type()?.is_dir() && validate_id(&name).is_ok() {
                folders.push(name);
            }
        }
        folders.sort();
        Ok(folders)
    }

    pub fn replay_games(&self, folder: &str) -> Result<Vec<String>> {
        validate_id(folder)?;
        let mut games = Vec::new();
        for entry in fs::read_dir(self.root.join("replays").join(folder))? {
            let entry = entry?;
            if entry.file_type()?.is_file()
                && let Some(id) = entry
                    .file_name()
                    .to_str()
                    .and_then(|name| name.strip_suffix(".json"))
                && validate_id(id).is_ok()
            {
                games.push(id.to_owned());
            }
        }
        games.sort();
        Ok(games)
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
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("parsing {}", path.display()))
}

pub fn read_toml<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// Never replace an existing experiment artifact, even when its ID was reused.
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        serde_json::to_writer_pretty(&mut writer, value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    temporary.as_file().sync_all()?;
    temporary
        .persist_noclobber(path)
        .with_context(|| format!("publishing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publication_is_atomic_and_never_overwrites() {
        struct Failure;
        impl Serialize for Failure {
            fn serialize<S: serde::Serializer>(
                &self,
                _: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("serialization failed"))
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifact.json");
        assert!(write_json(&path, &Failure).is_err());
        assert!(!path.exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
        write_json(&path, &42).unwrap();
        assert!(write_json(&path, &43).is_err());
        assert_eq!(std::fs::read_to_string(path).unwrap(), "42\n");
    }
}
