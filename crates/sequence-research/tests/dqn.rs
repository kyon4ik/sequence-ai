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

    let evaluate = Command::new(executable)
        .arg("evaluate-dqn")
        .arg(directory.join("checkpoints/episode-000001.bpk"))
        .arg(&legacy_path)
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

    legacy_config["device"] = json!("cuda");
    legacy_config["inference_device"] = json!("cpu");
    fs::write(&legacy_path, legacy_config.to_string()).unwrap();
    let evaluate_on_cpu = Command::new(executable)
        .arg("evaluate-dqn")
        .arg(directory.join("checkpoints/episode-000001.bpk"))
        .arg(&legacy_path)
        .output()
        .unwrap();
    assert!(
        evaluate_on_cpu.status.success(),
        "{}",
        String::from_utf8_lossy(&evaluate_on_cpu.stderr)
    );
    assert_eq!(evaluate_on_cpu.stdout, evaluate.stdout);
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
