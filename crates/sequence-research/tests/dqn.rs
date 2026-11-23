use std::{fs, process::Command};

use serde_json::{Value, json};

#[test]
fn trains_and_evaluates_a_checkpoint_against_both_baselines() {
    let workspace = tempfile::tempdir().unwrap();
    let config_path = workspace.path().join("dqn.json");
    let mut config: Value =
        serde_json::from_str(include_str!("../../../experiments/dqn.json")).unwrap();
    config["episodes"] = json!(1);
    config["action_limit"] = json!(8);
    config["batch_size"] = json!(2);
    config["replay_warmup"] = json!(2);
    config["train_every"] = json!(1);
    config["target_update_every"] = json!(1);
    config["evaluation"]["deal_count"] = json!(1);
    fs::write(&config_path, config.to_string()).unwrap();

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

    let evaluate = Command::new(executable)
        .arg("evaluate-dqn")
        .arg(directory.join("checkpoints/episode-000001.bpk"))
        .arg(directory.join("config.json"))
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
}
