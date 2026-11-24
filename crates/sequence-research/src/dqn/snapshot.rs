use std::{
    fs::File,
    io::{BufReader, BufWriter, ErrorKind, Read, Write},
    path::{Path, PathBuf},
};

use burn::tensor::Bytes;
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use crate::Result;

use super::{Config, training::Trainer};

const FILE_NAME: &str = "training-state.cbor";
const MAGIC: &[u8; 7] = b"SEQDQN\x01";

#[derive(Clone, Copy, Deserialize, Serialize)]
pub(super) struct LogOffsets {
    pub training: u64,
    pub evaluation: u64,
}

#[derive(Deserialize, Serialize)]
pub(super) struct Metadata {
    pub config: Config,
    pub episode: usize,
    pub logs: LogOffsets,
}

pub(super) fn policy_path(run: &Path, episode: usize) -> PathBuf {
    run.join("checkpoints")
        .join(format!("episode-{episode:06}.bpk"))
}

pub(super) fn save(
    run: &Path,
    metadata: &Metadata,
    trainer: &Trainer<'_>,
    online: &Bytes,
) -> Result<()> {
    atomic_write(&run.join(FILE_NAME), |writer| {
        writer.write_all(MAGIC)?;
        ciborium::into_writer(metadata, &mut *writer)?;
        trainer.save_state(writer, online)
    })
}

pub(super) fn open(run: &Path) -> Result<(Metadata, BufReader<File>)> {
    let path = run.join(FILE_NAME);
    let mut reader = BufReader::new(File::open(&path).map_err(|error| {
        if error.kind() == ErrorKind::NotFound {
            format!("{} has no resumable training snapshot", run.display())
        } else {
            format!("{}: {error}", path.display())
        }
    })?);
    let mut magic = [0; MAGIC.len()];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(format!("unsupported training snapshot format in {}", path.display()).into());
    }
    let metadata = ciborium::from_reader(&mut reader)?;
    Ok((metadata, reader))
}

pub(super) fn atomic_write(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<&mut File>) -> Result<()>,
) -> Result<()> {
    let directory = path.parent().expect("run artifact has a parent directory");
    let mut file = NamedTempFile::new_in(directory)?;
    let mut writer = BufWriter::new(file.as_file_mut());
    write(&mut writer)?;
    writer.flush()?;
    drop(writer);
    file.as_file().sync_all()?;
    file.persist(path)?;
    #[cfg(unix)]
    File::open(directory)?.sync_all()?;
    Ok(())
}

pub(super) mod array {
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};

    pub fn serialize<S, T, const N: usize>(array: &[T; N], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize,
    {
        array.as_slice().serialize(serializer)
    }

    pub fn deserialize<'de, D, T, const N: usize>(deserializer: D) -> Result<[T; N], D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        let values = Vec::<T>::deserialize(deserializer)?;
        values.try_into().map_err(|values: Vec<T>| {
            D::Error::custom(format!("expected {N} entries, found {}", values.len()))
        })
    }
}
