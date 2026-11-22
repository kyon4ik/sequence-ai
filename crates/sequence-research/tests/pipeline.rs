use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    process::{Child, Command, Stdio},
    time::Duration,
};

#[test]
fn config_reproduces_all_artifacts_when_both_policies_reach_the_limit() {
    let workspace = tempfile::tempdir().unwrap();
    let config = workspace.path().join("self-play.json");
    fs::write(
        &config,
        r#"{
            "policies": ["random", "random"],
            "first_shuffle_seed": 18446744073709551615,
            "deal_count": 1,
            "policy_seed": 42,
            "action_limit": 1
        }"#,
    )
    .unwrap();

    let executable = env!("CARGO_BIN_EXE_sequence-research");
    let first = workspace.path().join("first");
    let second = workspace.path().join("second");
    for directory in [&first, &second] {
        let run = Command::new(executable)
            .arg("run")
            .arg(&config)
            .arg("--output")
            .arg(directory)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
    }

    for artifact in [
        "config.json",
        "games.csv",
        "summary.json",
        "match-score.svg",
        "game-lengths.svg",
    ] {
        assert_eq!(
            fs::read(first.join(artifact)).unwrap(),
            fs::read(second.join(artifact)).unwrap(),
            "{artifact}"
        );
    }
    let summary: serde_json::Value =
        serde_json::from_slice(&fs::read(first.join("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["board_layout"], "tabletop");
    assert_eq!(summary["overall"]["truncations"], 2);
    assert!(summary["overall"]["score"].is_null());
    assert_eq!(
        summary["overall"]["score_bounds"],
        serde_json::json!([0.0, 1.0])
    );
    for field in ["paired_ci_95", "starts_ci_95", "moves_second_ci_95"] {
        assert_eq!(summary.get(field), Some(&serde_json::Value::Null));
    }

    for summary in ["{}", r#"{"board_layout":"concentric_rings"}"#] {
        fs::write(second.join("summary.json"), summary).unwrap();
        let legacy = Command::new(executable)
            .arg("inspect")
            .arg(&second)
            .args(["3", "--no-open"])
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&legacy.stderr);
        assert!(!legacy.status.success());
        assert!(
            stderr.contains("requires the tabletop board layout"),
            "{stderr}"
        );
    }

    let mut inspection = Viewer(
        Command::new(executable)
            .arg("inspect")
            .arg(&first)
            .args(["2", "--action", "1", "--no-open"])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let mut address = String::new();
    BufReader::new(inspection.0.stdout.take().unwrap())
        .read_line(&mut address)
        .unwrap();
    let address = address
        .strip_prefix("Replay viewer at http://")
        .and_then(|url| url.trim_end().strip_suffix("/?game=2&action=1"))
        .expect("the viewer should print its local address with the selected game and action");
    assert!(get(address, "/?game=2&action=1").contains("Sequence Replay Viewer"));

    let catalog = json_response(address, "/api/run");
    assert_eq!(catalog["policies"], serde_json::json!(["random", "random"]));
    let games = catalog["games"].as_array().unwrap();
    assert_eq!(games.len(), 2);
    assert_eq!(games[0]["policy_a_seat"], 0);
    assert_eq!(games[1]["policy_a_seat"], 1);
    assert_eq!(games[1]["shuffle_seed"], u64::MAX.to_string());
    assert_eq!(games[1]["outcome"], "action_limit");

    let first_game = json_response(address, "/api/games/1");
    assert_eq!(first_game["frames"][0]["active_player"], 0);
    let replay = json_response(address, "/api/games/2");
    assert_eq!(replay["board"][1], "10♠");
    assert_eq!(replay["board"][19], "6♦");
    let frames = replay["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0]["active_player"], 0);
    assert!(frames[1]["active_player"].is_null());
    let action = &replay["actions"][0];
    assert_eq!(action["player"], 0);
    assert_eq!(action["kind"], "play");
    assert!(
        frames[0]["hands"][0]
            .as_array()
            .unwrap()
            .iter()
            .any(|card| {
                card["card"] == action["card"]
                    && card["targets"]
                        .as_array()
                        .unwrap()
                        .contains(&action["target"])
            })
    );
    let target = action["target"].as_u64().unwrap() as usize;
    assert!(frames[0]["chips"][target].is_null());
    assert_eq!(frames[1]["chips"][target], 0);
    assert!(get(address, "/api/games/3").starts_with("HTTP/1.1 404"));
}

struct Viewer(Child);

impl Drop for Viewer {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn get(address: &str, path: &str) -> String {
    let mut stream = TcpStream::connect(address).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    response
}

fn json_response(address: &str, path: &str) -> serde_json::Value {
    let response = get(address, path);
    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
    assert!(headers.starts_with("HTTP/1.1 200"), "{response}");
    serde_json::from_str(body).unwrap()
}
