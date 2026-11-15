use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::{Value, json};

fn cli(data: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sequence"))
        .arg("--data-dir")
        .arg(data)
        .args(args)
        .output()
        .unwrap()
}

fn success(output: Output) -> Output {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn read(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn report_path(data: &Path) -> PathBuf {
    let mut runs: Vec<_> = fs::read_dir(data.join("runs"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    runs.sort();
    runs.last().unwrap().join("report.json")
}

#[test]
fn evaluation_persists_metrics_seeds_and_replayable_games() {
    let temp = tempfile::tempdir().unwrap();
    let data = temp.path().join("data");
    success(cli(
        &data,
        &["eval", "random", "greedy", "--games", "4", "--seed", "7"],
    ));
    let report = read(report_path(&data));
    assert_eq!(report["games"].as_array().unwrap().len(), 4);
    for (i, game) in report["games"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            game["players"],
            if i % 2 == 0 {
                json!([0, 1])
            } else {
                json!([1, 0])
            }
        );
        let id = game["replay"].as_str().unwrap();
        assert!(id.starts_with(&format!("{}/game-", report["id"].as_str().unwrap())));
        let record = read(data.join("replays").join(format!("{id}.json")));
        let output = success(cli(&data, &["replay", id, "--json"]));
        assert_eq!(
            record,
            serde_json::from_slice::<Value>(&output.stdout).unwrap()
        );
        success(cli(&data, &["replay", id, "--frame", "0"]));
    }
    for standing in report["standings"].as_array().unwrap() {
        let played: u64 = ["wins", "losses", "draws", "truncated"]
            .iter()
            .map(|key| standing[key].as_u64().unwrap())
            .sum();
        assert_eq!(played, 4);
        assert!(standing["decisions"].as_u64().unwrap() > 0);
    }
    let summary = success(cli(
        &data,
        &[
            "analyze",
            report["id"].as_str().unwrap(),
            "--format",
            "json",
        ],
    ));
    let summary: Value = serde_json::from_slice(&summary.stdout).unwrap();
    assert_eq!(summary["total_games"], 4);
    assert_eq!(summary["matrix"][0][1]["completed"], 4);
    assert_eq!(
        summary["matrix"][0][1]["wins"],
        report["standings"][0]["wins"]
    );
    assert_eq!(
        summary["matrix"][0][1]["losses"],
        summary["matrix"][1][0]["wins"]
    );
    assert_eq!(summary["matrix"][0][1]["as_red"]["games"], 2);
    assert_eq!(summary["matrix"][0][1]["as_blue"]["games"], 2);
    // Seeded actions and outcomes remain stable; timings are intentionally excluded.
    let other = temp.path().join("repeat");
    success(cli(
        &other,
        &["eval", "random", "greedy", "--games", "4", "--seed", "7"],
    ));
    let repeat = read(report_path(&other));
    for (a, b) in report["games"]
        .as_array()
        .unwrap()
        .iter()
        .zip(repeat["games"].as_array().unwrap())
    {
        assert_eq!(a["termination"], b["termination"]);
        let load = |root: &Path, g: &Value| {
            read(
                root.join("replays")
                    .join(format!("{}.json", g["replay"].as_str().unwrap())),
            )
        };
        let a = load(&data, a);
        let b = load(&other, b);
        assert_eq!(a["header"]["config"], b["header"]["config"]);
        let actions = |r: &Value| {
            r["steps"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s["action"].clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(actions(&a), actions(&b));
    }
}

#[test]
fn tournament_resolves_relative_specs_and_excludes_truncations_from_elo() {
    let temp = tempfile::tempdir().unwrap();
    let data = temp.path().join("data");
    let config_dir = temp.path().join("configs");
    fs::create_dir(&config_dir).unwrap();
    fs::write(
        config_dir.join("other.json"),
        serde_json::to_vec(&sequence_agent::AgentSpec {
            descriptor: sequence_agent::AgentDescriptor::new("other", "Other random"),
            config: sequence_agent::AgentConfig::Random,
        })
        .unwrap(),
    )
    .unwrap();
    let config = config_dir.join("league.toml");
    fs::write(
        &config,
        "agents = ['random', 'greedy', 'other.json']\ngames_per_pair = 2\nmax_actions = 1\n",
    )
    .unwrap();
    success(cli(
        &data,
        &["tournament", "--config", config.to_str().unwrap()],
    ));
    let report = read(report_path(&data));
    assert_eq!(report["games"].as_array().unwrap().len(), 6);
    for standing in report["standings"].as_array().unwrap() {
        assert_eq!(standing["truncated"], 4);
        assert_eq!(standing["draws"], 0);
        assert_eq!(standing["elo"], 1000.);
    }
    let path = report_path(&data);
    // Reports remain analyzable without any model, agent or replay files.
    let isolated = temp.path().join("report-only.json");
    fs::copy(path, &isolated).unwrap();
    let output = success(cli(
        &temp.path().join("no-data"),
        &["analyze", isolated.to_str().unwrap(), "--format", "json"],
    ));
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    let pair = &summary["matrix"][0][1];
    assert_eq!(pair["games"], 2);
    assert_eq!(pair["truncated"], 2);
    assert_eq!(pair["draws"], 0);
    assert!(pair["win_rate"].is_null());
    assert!(pair["score_rate"].is_null());
    let csv = success(cli(
        &data,
        &["analyze", report["id"].as_str().unwrap(), "--format", "csv"],
    ));
    assert_eq!(String::from_utf8(csv.stdout).unwrap().lines().count(), 7);
    fs::write(&config, "agents = ['greedy', 'heuristic']\n").unwrap();
    assert!(
        !cli(&data, &["tournament", "--config", config.to_str().unwrap()])
            .status
            .success()
    );
}

#[test]
fn errors_are_nonzero_and_replay_corruption_is_detected() {
    let temp = tempfile::tempdir().unwrap();
    let data = temp.path().join("data");
    assert!(!cli(&data, &["eval", "missing", "random"]).status.success());
    assert!(!data.exists());
    success(cli(
        &data,
        &[
            "eval",
            "random",
            "random",
            "--games",
            "1",
            "--max-actions",
            "1",
        ],
    ));
    let report = read(report_path(&data));
    let id = report["games"][0]["replay"].as_str().unwrap();
    assert!(!cli(&data, &["replay", id, "--frame", "2"]).status.success());
    let path = data.join("replays").join(format!("{id}.json"));
    fs::copy(&path, data.join("replays/legacy.json")).unwrap();
    let flat = cli(&data, &["replay", "legacy"]);
    assert!(!flat.status.success());
    assert!(String::from_utf8_lossy(&flat.stderr).contains("folder/game"));
    let mut record = read(&path);
    record["steps"][0]["actor"] = json!("Blue");
    fs::write(path, serde_json::to_vec(&record).unwrap()).unwrap();
    assert!(!cli(&data, &["replay", id]).status.success());
}

#[cfg(feature = "burn")]
fn train_save_reload_and_evaluate(device: &str) {
    let temp = tempfile::tempdir().unwrap();
    let data = temp.path().join("data");
    let config = temp.path().join("mc.toml");
    fs::write(&config, "algorithm = 'mc'\nid = 'test-model'\n[training]\niterations = 1\ngames_per_iteration = 1\nbatch_size = 128\nevaluation_games = 2\nepsilon_start = 1.0\nepsilon_end = 1.0\nseed = 7\n").unwrap();
    success(cli(
        &data,
        &[
            "train",
            "--config",
            config.to_str().unwrap(),
            "--device",
            device,
        ],
    ));
    let model = data.join("models/test-model");
    assert!(model.join("model.bpk").is_file());
    assert!(model.join("best.bpk").is_file());
    assert!(
        model
            .join("checkpoints/iteration-000001/optimizer.bpk")
            .is_file()
    );
    assert!(
        model
            .join("checkpoints/iteration-000001/model.bpk")
            .is_file()
    );
    assert!(
        model
            .join("checkpoints/iteration-000001/state.json")
            .is_file()
    );
    let score = read(model.join("report.json"))["iterations"][0]["evaluation_score"]
        .as_f64()
        .unwrap();
    assert!((0.0..=1.0).contains(&score));
    let spec = read(model.join("agent.json"));
    assert_eq!(spec["config"]["checkpoint"], "model.bpk");
    assert_eq!(
        read(model.join("report.json"))["iterations"][0]["episodes"],
        1
    );
    success(cli(
        &data,
        &[
            "train",
            "--config",
            config.to_str().unwrap(),
            "--device",
            device,
            "--id",
            "resumed-model",
            "--resume",
            model.join("checkpoints/iteration-000001").to_str().unwrap(),
        ],
    ));
    assert_eq!(
        read(model.join("report.json")),
        read(data.join("models/resumed-model/report.json"))
    );
    assert!(data.join("models/resumed-model/best.bpk").is_file());
    let before = fs::read(model.join("model.bpk")).unwrap();
    assert!(
        !cli(
            &data,
            &[
                "train",
                "--config",
                config.to_str().unwrap(),
                "--device",
                device
            ]
        )
        .status
        .success()
    );
    assert_eq!(before, fs::read(model.join("model.bpk")).unwrap());
    // Resolve the model by ID and by portable spec path, using the same agent factory.
    success(cli(
        &data,
        &[
            "eval",
            "test-model",
            "heuristic",
            "--games",
            "1",
            "--max-actions",
            "2",
            "--device",
            device,
        ],
    ));
    success(cli(
        &data,
        &[
            "eval",
            model.join("agent.json").to_str().unwrap(),
            "random",
            "--games",
            "1",
            "--max-actions",
            "2",
            "--device",
            device,
        ],
    ));
    let report = read(report_path(&data));
    success(cli(
        &data,
        &[
            "replay",
            report["games"][0]["replay"].as_str().unwrap(),
            "--json",
        ],
    ));
}

#[cfg(feature = "burn")]
#[test]
fn cpu_training_cli_roundtrip() {
    train_save_reload_and_evaluate("cpu");
}

#[cfg(feature = "burn")]
#[test]
#[ignore = "requires host NVIDIA GPU access; run outside the sandbox"]
fn cuda_training_cli_roundtrip() {
    train_save_reload_and_evaluate("cuda:0");
}

#[cfg(feature = "burn")]
#[test]
fn invalid_training_configs_fail_before_creating_artifacts() {
    let temp = tempfile::tempdir().unwrap();
    let data = temp.path().join("data");
    let config = temp.path().join("bad.toml");
    for text in [
        "algorithm = 'ppo'\nid = 'x'",
        "algorithm = 'mc'\nid = 'x'\n[training]\niterations = 0",
        "algorithm = 'mc'\nid = 'x'\n[training]\ninference_batch_games = 0",
        "algorithm = 'mc'\nid = 'x'\n[training]\nbatch_szie = 1",
        "algorithm = 'mc'\nid = 'x'\n[training]\nlearning_rate = nan",
        "algorithm = 'mc'\nid = 'x'\n[training]\nupdates_per_iteration = 0",
        "algorithm = 'mc'\nid = 'x'\n[training]\nsampling = 'unknown'",
        "algorithm = 'mc'\nid = 'x'\n[training.opponents]\ngreedy = -1",
        "algorithm = 'mc'\nid = 'x'\n[training.opponents]\nself_play = 0",
        "algorithm = 'mc'\nid = 'x'\n[training.opponents]\ngreedy = nan",
        "algorithm = 'mc'\nid = 'x'\n[training.opponents]\ngreddy = 10",
        "algorithm = 'mc'\nid = '../x'",
    ] {
        fs::write(&config, text).unwrap();
        assert!(
            !cli(&data, &["train", "--config", config.to_str().unwrap()])
                .status
                .success()
        );
        assert!(!data.exists());
    }
}

#[cfg(not(feature = "burn"))]
#[test]
fn training_explains_missing_feature() {
    let temp = tempfile::tempdir().unwrap();
    let output = cli(temp.path(), &["train", "--config", "unused.toml"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Burn support"));
}
