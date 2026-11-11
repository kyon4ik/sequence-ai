use std::{
    fs,
    net::{IpAddr, SocketAddr},
    path::Path,
};

use anyhow::{Result, bail};
use sequence_core::{Position, get_card};
use serde_json::{Value, json};
use tiny_http::{Header, Method, Response, Server, StatusCode};

use crate::{
    artifacts::{Store, read_json},
    validate_id,
};

pub fn serve(store: &Store, host: IpAddr, port: u16) -> Result<()> {
    let server = Server::http(SocketAddr::new(host, port))
        .map_err(|e| anyhow::anyhow!("starting replay server: {e}"))?;
    println!(
        "Replay explorer: http://{} (data: {})",
        server.server_addr(),
        store.root.display()
    );
    for request in server.incoming_requests() {
        let (status, mime, body) = if *request.method() != Method::Get {
            (
                405,
                "text/plain; charset=utf-8",
                b"Only GET is supported".to_vec(),
            )
        } else {
            match route(store, request.url().split('?').next().unwrap_or("/")) {
                Ok(response) => response,
                Err(error) => (
                    400,
                    "text/plain; charset=utf-8",
                    format!("{error:#}").into_bytes(),
                ),
            }
        };
        let response = Response::from_data(body)
            .with_status_code(StatusCode(status))
            .with_header(Header::from_bytes("Content-Type", mime).unwrap())
            .with_header(Header::from_bytes("Cache-Control", "no-store").unwrap());
        if let Err(error) = request.respond(response) {
            eprintln!("serving request: {error}");
        }
    }
    Ok(())
}

type Reply = (u16, &'static str, Vec<u8>);

fn json_reply(value: impl serde::Serialize) -> Result<Reply> {
    Ok((200, "application/json", serde_json::to_vec(&value)?))
}

fn entries(path: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    if !path.exists() {
        return Ok(names);
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if let Some(name) = entry.file_name().to_str() {
            names.push(name.to_owned());
        }
    }
    names.sort();
    Ok(names)
}

fn json_file(path: &Path) -> Result<Reply> {
    if !path.is_file() {
        return Ok((404, "text/plain", b"Not found".to_vec()));
    }
    json_reply(read_json::<Value>(path)?)
}

fn route(store: &Store, path: &str) -> Result<Reply> {
    match path {
        "/" => Ok((
            200,
            "text/html; charset=utf-8",
            include_bytes!("replay.html").to_vec(),
        )),
        "/api/board" => json_reply(
            Position::all()
                .map(|p| {
                    get_card(p)
                        .map(crate::replay::card_label)
                        .unwrap_or_else(|| "★".into())
                })
                .collect::<Vec<_>>(),
        ),
        "/api/models" => {
            let mut models = Vec::new();
            for id in entries(&store.root.join("models"))? {
                validate_id(&id)?;
                let dir = store.model_dir(&id)?;
                if dir.join("agent.json").is_file() {
                    models.push(json!({"id": id, "agent": read_json::<Value>(&dir.join("agent.json"))?, "training": if dir.join("report.json").is_file() { read_json::<Value>(&dir.join("report.json"))? } else { Value::Null }}));
                }
            }
            json_reply(models)
        }
        "/api/replays" => json_reply(store.replay_folders()?),
        "/api/runs" => json_reply(
            entries(&store.root.join("runs"))?
                .into_iter()
                .filter(|id| {
                    validate_id(id).is_ok()
                        && store
                            .root
                            .join("runs")
                            .join(id)
                            .join("report.json")
                            .is_file()
                })
                .collect::<Vec<_>>(),
        ),
        _ => {
            if let Some(id) = path.strip_prefix("/api/replays/") {
                if id.contains('/') {
                    return json_file(&store.replay_path(id)?);
                }
                validate_id(id)?;
                if !store.root.join("replays").join(id).is_dir() {
                    return Ok((404, "text/plain", b"Not found".to_vec()));
                }
                return json_reply(store.replay_games(id)?);
            }
            if let Some(id) = path.strip_prefix("/api/runs/") {
                validate_id(id)?;
                return json_file(&store.root.join("runs").join(id).join("report.json"));
            }
            if path.starts_with("/api/") {
                bail!("unknown API route");
            }
            Ok((404, "text/plain", b"Not found".to_vec()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replays_are_browsed_by_folder_and_flat_files_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path().to_path_buf());
        fs::create_dir_all(dir.path().join("replays/run-1")).unwrap();
        fs::create_dir_all(dir.path().join("replays/run-2")).unwrap();
        fs::create_dir_all(dir.path().join("replays/invalid.folder")).unwrap();
        fs::create_dir_all(dir.path().join("replays/run-1/nested.json")).unwrap();
        fs::write(dir.path().join("replays/run-1/game-2.json"), "{}").unwrap();
        fs::write(dir.path().join("replays/run-1/game-1.json"), "{}").unwrap();
        fs::write(dir.path().join("replays/run-1/invalid.game.json"), "{}").unwrap();
        fs::write(dir.path().join("replays/run-1/report.txt"), "{}").unwrap();
        fs::write(dir.path().join("replays/legacy.json"), "{}").unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&route(&store, "/api/replays").unwrap().2).unwrap(),
            json!(["run-1", "run-2"])
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&route(&store, "/api/replays/run-1").unwrap().2)
                .unwrap(),
            json!(["game-1", "game-2"])
        );
        assert_eq!(route(&store, "/api/replays/run-2").unwrap().2, b"[]");
        assert_eq!(route(&store, "/api/replays/run-1/game-1").unwrap().0, 200);
        assert_eq!(route(&store, "/api/replays/legacy").unwrap().0, 404);
        assert_eq!(route(&store, "/api/replays/run-1/missing").unwrap().0, 404);
        for id in [
            "legacy",
            "../secret",
            "run/../secret",
            "run//game",
            "/game",
            "run/%2e%2e",
        ] {
            assert!(store.replay_path(id).is_err());
        }
        for path in [
            "/api/replays/..",
            "/api/replays/../secret",
            "/api/replays/run/../secret",
            "/api/replays/run//game",
            "/api/replays//game",
            "/api/replays/run/%2e%2e",
        ] {
            assert!(route(&store, path).is_err());
        }
    }

    #[test]
    fn empty_store_is_viewable_and_routes_reject_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(dir.path().join("missing"));
        for path in ["/api/models", "/api/replays", "/api/runs"] {
            let (status, _, body) = route(&store, path).unwrap();
            assert_eq!(status, 200);
            assert_eq!(body, b"[]");
        }
        assert!(route(&store, "/api/replays/../../secret").is_err());
        assert!(route(&store, "/api/runs/%2e%2e").is_err());
        assert_eq!(route(&store, "/api/replays/missing").unwrap().0, 404);
        assert_eq!(
            serde_json::from_slice::<Vec<String>>(&route(&store, "/api/board").unwrap().2)
                .unwrap()
                .len(),
            100
        );
    }
}
