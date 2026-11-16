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
    let path = path.as_ref();
    if path.extension().is_some_and(|ext| ext == "gz") {
        serde_json::from_reader(flate2::read::GzDecoder::new(fs::File::open(path).unwrap()))
            .unwrap()
    } else {
        serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
    }
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
fn evaluation_defaults_to_results_and_seeds_without_replays() {
    let temp = tempfile::tempdir().unwrap();
    success(cli(
        temp.path(),
        &[
            "eval",
            "random",
            "greedy",
            "--games",
            "2",
            "--max-actions",
            "1",
        ],
    ));
    assert!(!temp.path().join("replays").exists());
    let report = read(report_path(temp.path()));
    for game in report["games"].as_array().unwrap() {
        assert!(game["replay"].is_null());
        assert!(game["config"]["game_seed"].is_u64());
        assert_eq!(game["config"]["agent_seeds"].as_array().unwrap().len(), 2);
    }
    success(cli(
        temp.path(),
        &["analyze", report["id"].as_str().unwrap()],
    ));
}

#[test]
fn evaluation_persists_metrics_seeds_and_replayable_games() {
    let temp = tempfile::tempdir().unwrap();
    let data = temp.path().join("data");
    success(cli(
        &data,
        &[
            "eval",
            "--save-replays",
            "random",
            "greedy",
            "--games",
            "4",
            "--seed",
            "7",
        ],
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
        let record = read(data.join("replays").join(format!("{id}.json.gz")));
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
        &[
            "eval",
            "--save-replays",
            "random",
            "greedy",
            "--games",
            "4",
            "--seed",
            "7",
        ],
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
                    .join(format!("{}.json.gz", g["replay"].as_str().unwrap())),
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
    fs::write(&config, "agents = ['greedy', 'greedy']\n").unwrap();
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
    assert!(
        !cli(&data, &["eval", "--save-replays", "missing", "random"])
            .status
            .success()
    );
    assert!(
        !cli(&data, &["eval", "--save-replays", "heuristic", "random"])
            .status
            .success()
    );
    assert!(!data.exists());
    success(cli(
        &data,
        &[
            "eval",
            "--save-replays",
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
    let path = data.join("replays").join(format!("{id}.json.gz"));
    fs::copy(&path, data.join("replays/legacy.json")).unwrap();
    let flat = cli(&data, &["replay", "legacy"]);
    assert!(!flat.status.success());
    assert!(String::from_utf8_lossy(&flat.stderr).contains("folder/game"));
    let mut record = read(&path);
    record["steps"][0]["actor"] = json!("Blue");
    let mut encoder = flate2::write::GzEncoder::new(
        fs::File::create(path).unwrap(),
        flate2::Compression::default(),
    );
    serde_json::to_writer(&mut encoder, &record).unwrap();
    encoder.finish().unwrap();
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
            "--save-replays",
            "test-model",
            "greedy",
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
            "--save-replays",
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

fn digest(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(fs::read(path).unwrap()))
}

fn gate_fixture(root: &Path, name: &str, score: f64) -> (PathBuf, PathBuf) {
    let model = root.join(format!("{name}.bpk"));
    fs::write(&model, name).unwrap();
    let spec = root.join(format!("{name}-agent.json"));
    fs::write(&spec, serde_json::to_vec(&json!({"descriptor":{"id":name,"name":name,"metadata":{}},"config":{"kind":"monte_carlo","checkpoint":model.file_name().unwrap().to_str().unwrap()}})).unwrap()).unwrap();
    let games: Vec<_> = (0..200).flat_map(|seed| [
        json!({"config":{"game_seed":seed,"agent_seeds":[1,2],"max_actions":1000},"learner_seat":0,"score":score}),
        json!({"config":{"game_seed":seed,"agent_seeds":[2,1],"max_actions":1000},"learner_seat":1,"score":score}),
    ]).collect();
    let report = root.join(format!("{name}-report.json"));
    fs::write(&report,serde_json::to_vec(&json!({"checkpoint":model,"checkpoint_sha256":digest(&model),"device":"cpu",
        "config":{"games":400,"seed":7,"inference_batch_games":128},"baselines":[{
        "baseline":{"kind":"greedy","weights":[0,0,1,5,10,100]},"summary":{"opponent":"greedy","games":400,"score":score},"games":games}]})).unwrap()).unwrap();
    (report, spec)
}

#[test]
fn promotion_is_bound_idempotent_and_preserves_champion_on_failure() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let (old, old_spec) = gate_fixture(root, "old", 0.);
    let (better, better_spec) = gate_fixture(root, "better", 0.5);
    let (worse, worse_spec) = gate_fixture(root, "worse", 0.);
    let registry = root.join("champion.json");
    let decision = root.join("promoted.json");
    let run = |incumbent: &Path,
               candidate: &Path,
               incumbent_spec: &Path,
               candidate_spec: &Path,
               decision: &Path,
               extra: &[&str]| {
        let mut args = vec![
            "promote",
            "--incumbent",
            incumbent.to_str().unwrap(),
            "--candidate",
            candidate.to_str().unwrap(),
            "--incumbent-spec",
            incumbent_spec.to_str().unwrap(),
            "--candidate-spec",
            candidate_spec.to_str().unwrap(),
            "--registry",
            registry.to_str().unwrap(),
            "--decision",
            decision.to_str().unwrap(),
        ];
        args.extend_from_slice(extra);
        cli(root, &args)
    };
    success(run(&old, &better, &old_spec, &better_spec, &decision, &[]));
    let bytes = fs::read(&registry).unwrap();
    assert_eq!(
        read(&registry)["config"]["checkpoint"],
        root.join("better.bpk").to_str().unwrap()
    );
    assert_eq!(read(&registry)["descriptor"]["id"], "champion");
    success(run(&old, &better, &old_spec, &better_spec, &decision, &[]));
    assert_eq!(fs::read(&registry).unwrap(), bytes);
    assert!(
        !run(
            &old,
            &better,
            &old_spec,
            &better_spec,
            &decision,
            &["--family-alpha", "0.025"]
        )
        .status
        .success()
    );
    let failed = root.join("failed.json");
    assert!(
        !run(&better, &worse, &better_spec, &worse_spec, &failed, &[])
            .status
            .success()
    );
    assert_eq!(read(&failed)["passed"], false);
    assert_eq!(fs::read(&registry).unwrap(), bytes);
    fs::write(root.join("worse.bpk"), "changed").unwrap();
    assert!(
        !run(
            &better,
            &worse,
            &better_spec,
            &worse_spec,
            &root.join("changed.json"),
            &[]
        )
        .status
        .success()
    );
    assert!(!root.join("changed.json").exists());
    assert_eq!(fs::read(&registry).unwrap(), bytes);
    let confirmation = root.join("confirmation.json");
    fs::write(&confirmation, r#"{"confirmed":false}"#).unwrap();
    let blocked = root.join("blocked.json");
    let (perfect, perfect_spec) = gate_fixture(root, "perfect", 1.);
    assert!(
        !run(
            &better,
            &perfect,
            &better_spec,
            &perfect_spec,
            &blocked,
            &["--confirmation", confirmation.to_str().unwrap()]
        )
        .status
        .success()
    );
    assert_eq!(read(&blocked)["aggregate_confirmation_passed"], false);
    assert_eq!(fs::read(&registry).unwrap(), bytes);
    let frozen = root.join("frozen.bpk");
    fs::write(&frozen, "frozen opponent").unwrap();
    for report in [&better, &perfect] {
        let mut value = read(report);
        value["baselines"][0]["baseline"] = json!({"kind":"monte_carlo","checkpoint":frozen});
        fs::write(report, serde_json::to_vec(&value).unwrap()).unwrap();
    }
    let bound = root.join("bound.json");
    assert!(
        !run(&better, &perfect, &better_spec, &perfect_spec, &bound, &[])
            .status
            .success()
    );
    assert!(!bound.exists());
    assert_eq!(fs::read(&registry).unwrap(), bytes);
    for report in [&better, &perfect] {
        let mut value = read(report);
        value["opponent_checkpoint_sha256"] = json!({"greedy":digest(&frozen)});
        fs::write(report, serde_json::to_vec(&value).unwrap()).unwrap();
    }
    success(run(
        &better,
        &perfect,
        &better_spec,
        &perfect_spec,
        &bound,
        &[],
    ));
    let promoted = fs::read(&registry).unwrap();
    fs::write(&frozen, "changed opponent").unwrap();
    assert!(
        !run(&better, &perfect, &better_spec, &perfect_spec, &bound, &[])
            .status
            .success()
    );
    assert_eq!(fs::read(&registry).unwrap(), promoted);
}

#[cfg(feature = "burn")]
#[test]
fn training_intervals_and_panel_publish_bound_paired_evidence() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let data = root.join("data");
    let config = root.join("train.toml");
    fs::write(&config,"algorithm = 'mc'\nid = 'panel-model'\n[training]\niterations = 3\ngames_per_iteration = 1\nevaluation_games = 2\nepsilon_start = 1.0\nepsilon_end = 1.0\n").unwrap();
    success(cli(
        &data,
        &[
            "train",
            "--config",
            config.to_str().unwrap(),
            "--seed",
            "91",
            "--checkpoint-interval",
            "2",
        ],
    ));
    let model = data.join("models/panel-model");
    assert!(!model.join("checkpoints/iteration-000001").exists());
    assert!(
        model
            .join("checkpoints/iteration-000002/state.json")
            .is_file()
    );
    assert!(
        model
            .join("checkpoints/iteration-000003/state.json")
            .is_file()
    );
    assert_eq!(read(model.join("experiment.json"))["training"]["seed"], 91);
    assert_eq!(read(model.join("progress.json"))["iteration"], 3);
    let manifest = root.join("panel.json");
    let output = root.join("evaluations");
    let panel = json!({"device":"cuda:0","games":4,"inference_batch_games":2,"seeds":[77],"checkpoints":[
        {"id":"final","path":model.join("model.bpk")},{"id":"best","path":model.join("best.bpk")}],
        "opponents":[["greedy",{"kind":"greedy","weights":[0,0,1,5,10,100]}],["frozen",{"kind":"monte_carlo","checkpoint":model.join("best.bpk")}]],"diagnostic_games":2});
    fs::write(&manifest, serde_json::to_vec(&panel).unwrap()).unwrap();
    success(cli(
        &data,
        &[
            "panel",
            "--config",
            manifest.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--device",
            "cpu",
        ],
    ));
    let final_path = output.join("final-77.json");
    let bytes = fs::read(&final_path).unwrap();
    let final_report = read(&final_path);
    let best_report = read(output.join("best-77.json"));
    assert_eq!(final_report["device"], "cpu");
    assert_eq!(
        final_report["checkpoint_sha256"],
        digest(&model.join("model.bpk"))
    );
    assert_eq!(
        final_report["opponent_checkpoint_sha256"]["frozen"],
        digest(&model.join("best.bpk"))
    );
    let keys = |r: &Value| {
        r["baselines"][0]["games"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| (g["config"].clone(), g["learner_seat"].clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(keys(&final_report), keys(&best_report));
    assert_eq!(
        final_report["baselines"][0]["games"][0]["config"]["game_seed"],
        final_report["baselines"][0]["games"][1]["config"]["game_seed"]
    );
    assert!(
        !cli(
            &data,
            &[
                "panel",
                "--config",
                manifest.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
                "--device",
                "cpu"
            ]
        )
        .status
        .success()
    );
    assert_eq!(fs::read(&final_path).unwrap(), bytes);
    let mut invalid = panel;
    invalid["seeds"] = json!([77, 77]);
    fs::write(&manifest, serde_json::to_vec(&invalid).unwrap()).unwrap();
    let absent = root.join("invalid");
    assert!(
        !cli(
            &data,
            &[
                "panel",
                "--config",
                manifest.to_str().unwrap(),
                "--output",
                absent.to_str().unwrap(),
                "--device",
                "cpu"
            ]
        )
        .status
        .success()
    );
    assert!(!absent.exists());
}
