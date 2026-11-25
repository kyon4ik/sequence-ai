use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    process::Command,
};

use serde_json::{Value, json};

#[test]
fn trains_and_evaluates_a_checkpoint_against_both_baselines() {
    let workspace = tempfile::tempdir().unwrap();
    let config_path = workspace.path().join("dqn.json");
    fs::write(&config_path, include_str!("fixtures/dqn.json")).unwrap();

    let directory = workspace.path().join("run");
    let executable = env!("CARGO_BIN_EXE_sequence-research");
    let training = Command::new(executable)
        .arg("train-dqn")
        .arg(&config_path)
        .arg("--output")
        .arg(&directory)
        .output()
        .unwrap();
    assert!(
        training.status.success(),
        "{}",
        String::from_utf8_lossy(&training.stderr)
    );

    let mut reader = csv::Reader::from_path(directory.join("evaluation.csv")).unwrap();
    let rows: Vec<_> = reader.records().collect::<Result<_, _>>().unwrap();
    assert_eq!(rows.len(), 4);
    assert_eq!(&rows[2][3], "random");
    assert_eq!(&rows[3][3], "heuristic");
    assert!(rows[2][2].parse::<usize>().unwrap() > 0);
    let summary: Value =
        serde_json::from_slice(&fs::read(directory.join("summary.json")).unwrap()).unwrap();
    assert_eq!(
        summary["model"],
        json!({"trainable_parameters": 518_544, "weights": 517_792, "biases": 752})
    );
    let latest = summary["latest_evaluation"].as_array().unwrap();
    assert_eq!(latest.len(), 2);
    assert!(latest.iter().all(|row| row["episode"] == 1));
    assert!(
        fs::read_to_string(directory.join("win-rates.svg"))
            .unwrap()
            .contains("518544 trainable parameters")
    );

    let mut legacy_config: Value =
        serde_json::from_slice(&fs::read(directory.join("config.json")).unwrap()).unwrap();
    legacy_config["sequence_shaping_weight"] = json!(0.1);
    legacy_config["replay_curriculum"] =
        json!([{ "through_episode": 1, "max_decisions_to_end": 8 }]);
    let legacy_path = workspace.path().join("legacy.json");
    fs::write(&legacy_path, legacy_config.to_string()).unwrap();

    let report = workspace.path().join("checkpoint-report");
    let evaluate = Command::new(executable)
        .arg("evaluate-dqn")
        .arg(directory.join("checkpoints/episode-000001.bpk"))
        .arg(&legacy_path)
        .arg("--output")
        .arg(&report)
        .output()
        .unwrap();
    assert!(
        evaluate.status.success(),
        "{}",
        String::from_utf8_lossy(&evaluate.stderr)
    );
    let results: Value = serde_json::from_slice(&evaluate.stdout).unwrap();
    assert_eq!(results[0]["opponent"], "random");
    assert_eq!(results[1]["opponent"], "heuristic");
    assert_eq!(results[0]["cutoffs"], 2);
    assert_eq!(results[1]["cutoffs"], 2);
    let summary: Value =
        serde_json::from_slice(&fs::read(report.join("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["checkpoints"][0]["evaluation"], results);
    assert_eq!(summary["checkpoints"][0]["tactics"][0]["positions"], 0);
    assert!(
        fs::read_to_string(report.join("tactical-positions.csv"))
            .unwrap()
            .starts_with("run,episode,shuffle_seed,")
    );
    assert!(
        fs::read_to_string(report.join("win-rates.svg"))
            .unwrap()
            .contains("Learner decisions")
    );
    assert!(
        fs::read_to_string(report.join("tactics.svg"))
            .unwrap()
            .contains("0 positions")
    );

    legacy_config["device"] = json!("cuda");
    legacy_config["inference_device"] = json!("cpu");
    fs::write(&legacy_path, legacy_config.to_string()).unwrap();
    let evaluate_on_cpu = Command::new(executable)
        .arg("evaluate-dqn")
        .arg(directory.join("checkpoints/episode-000001.bpk"))
        .arg(&legacy_path)
        .arg("--output")
        .arg(workspace.path().join("cpu-report"))
        .output()
        .unwrap();
    assert!(
        evaluate_on_cpu.status.success(),
        "{}",
        String::from_utf8_lossy(&evaluate_on_cpu.stderr)
    );
    assert_eq!(evaluate_on_cpu.stdout, evaluate.stdout);

    legacy_config["action_limit"] = json!(256);
    legacy_config["tactics"] =
        json!({"first_shuffle_seed": 16000000, "deal_count": 1, "policy_seed": 2045});
    fs::write(&legacy_path, legacy_config.to_string()).unwrap();
    let run_report = workspace.path().join("run-report");
    let games = workspace.path().join("games.csv");
    run_successfully(
        Command::new(executable)
            .arg("evaluate-dqn")
            .arg(&directory)
            .arg(&legacy_path)
            .arg("--output")
            .arg(&run_report)
            .arg("--games")
            .arg(&games),
    );
    let mut tactics = csv::Reader::from_path(run_report.join("tactical-positions.csv")).unwrap();
    let scores: Vec<_> = tactics.records().collect::<Result<_, _>>().unwrap();
    assert!(!scores.is_empty());
    let (initial, final_checkpoint) = scores.split_at(scores.len() / 2);
    for (before, after) in initial.iter().zip(final_checkpoint) {
        assert_eq!(&before[1], "0");
        assert_eq!(&after[1], "1");
        assert_eq!(&after[2], "16000000");
        assert!(
            before
                .iter()
                .skip(2)
                .take(6)
                .eq(after.iter().skip(2).take(6))
        );
    }
    assert_eq!(initial.len(), final_checkpoint.len());
    let match_data = fs::read_to_string(games).unwrap();
    assert!(match_data.starts_with("episode,shuffle_seed,learner_seat,opponent,outcome\n"));
    assert_eq!(match_data.lines().count(), 9);
    assert!(
        fs::read_to_string(run_report.join("tactics.svg"))
            .unwrap()
            .contains("Tactical action ranked first")
    );

    let config = workspace.path().join("search.json");
    fs::write(
        &config,
        json!({
            "policies": ["ismcts", "dqn"],
            "checkpoint": "run/checkpoints/episode-000001.bpk",
            "search": {"simulations": 4, "depth": 2},
            "first_shuffle_seed": 18000000,
            "deal_count": 1,
            "policy_seed": 2080,
            "action_limit": 4
        })
        .to_string(),
    )
    .unwrap();

    let search_report = workspace.path().join("search-report");
    let repeated = workspace.path().join("search-repeated");
    for output in [&search_report, &repeated] {
        run_successfully(
            Command::new(executable)
                .arg("run")
                .arg(&config)
                .arg("--output")
                .arg(output),
        );
    }

    let games = fs::read(search_report.join("games.csv")).unwrap();
    assert_eq!(games, fs::read(repeated.join("games.csv")).unwrap());
    let times: Value =
        serde_json::from_slice(&fs::read(search_report.join("move-times.json")).unwrap()).unwrap();
    assert_eq!(times[0]["policy"], "ismcts");
    assert_eq!(times[0]["decisions"], 4);
    assert_eq!(times[1]["decisions"], 4);
}

#[test]
fn resume_recovers_the_latest_checkpoint_and_matches_uninterrupted_training() {
    let workspace = tempfile::tempdir().unwrap();
    let root = workspace.path();
    let config_path = root.join("config.json");
    let uninterrupted = root.join("uninterrupted");
    let resumed = root.join("resumed");
    let mut config: Value = serde_json::from_str(include_str!("fixtures/dqn.json")).unwrap();
    config["target"] = json!("double_dqn");
    let fields = config.as_object_mut().unwrap();
    for name in ["epsilon_start", "epsilon_end", "epsilon_decay_decisions"] {
        fields.remove(name);
    }
    config["temperature"] = json!({"start": 0.5, "end": 0.025, "decay_decisions": 80});
    config["checkpoint_every"] = json!(2);
    config["action_limit"] = json!(16);
    config["replay_capacity"] = json!(12);
    config["train_every"] = json!(2);
    config["target_update_every"] = json!(5);
    config["opponent_update_every"] = json!(4);
    config["heuristic_opponent_every"] = json!(5);

    let mut train = |episodes, output: &Path| {
        config["episodes"] = json!(episodes);
        fs::write(&config_path, config.to_string()).unwrap();
        run_successfully(
            Command::new(env!("CARGO_BIN_EXE_sequence-research"))
                .arg("train-dqn")
                .arg(&config_path)
                .arg("--output")
                .arg(output),
        );
    };
    train(8, &uninterrupted);
    train(6, &resumed);

    for name in ["training.csv", "evaluation.csv"] {
        let mut log = OpenOptions::new()
            .append(true)
            .open(resumed.join(name))
            .unwrap();
        log.write_all(b"incomplete trailing row").unwrap();
    }
    let split_checkpoint = resumed.join("checkpoints/episode-000006.bpk");
    let split_weights = fs::read(&split_checkpoint).unwrap();
    fs::remove_file(&split_checkpoint).unwrap();

    let state = resumed.join("training-state.cbor");
    let split_state = fs::read(&state).unwrap();
    run_successfully(
        Command::new(env!("CARGO_BIN_EXE_sequence-research"))
            .arg("resume-dqn")
            .arg(&resumed)
            .args(["--episodes", "8"]),
    );

    assert_eq!(fs::read(&split_checkpoint).unwrap(), split_weights);
    assert_ne!(fs::read(state).unwrap(), split_state);
    for name in [
        "training.csv",
        "evaluation.csv",
        "summary.json",
        "config.json",
    ] {
        assert_eq!(
            fs::read(resumed.join(name)).unwrap(),
            fs::read(uninterrupted.join(name)).unwrap(),
            "{name} differs after resuming",
        );
    }

    let final_tensors = |run: &Path| {
        burn_pack::Reader::from_file(run.join("checkpoints/episode-000008.bpk"))
            .unwrap()
            .into_tensors()
            .unwrap()
    };
    let expected = final_tensors(&uninterrupted);
    let actual = final_tensors(&resumed);
    assert_eq!(expected.len(), actual.len());
    for (expected, actual) in expected.into_iter().zip(actual) {
        let (name, _, _, _, expected) = expected.into_parts().unwrap();
        let (actual_name, _, _, _, actual) = actual.into_parts().unwrap();
        assert_eq!(name, actual_name);
        assert!(*expected == *actual, "tensor {name} differs after resuming");
    }
}

fn run_successfully(command: &mut Command) {
    let result = command.env("RAYON_NUM_THREADS", "2").output().unwrap();
    assert!(
        result.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}
