use std::{fs, process::Command};

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
            "starting_seat": "two",
            "action_limit": 1,
            "bootstrap": {"seed": 42, "samples": 100}
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
    assert_eq!(summary["overall"]["truncations"], 2);
    assert!(summary["overall"]["score"].is_null());
    assert_eq!(
        summary["overall"]["score_bounds"],
        serde_json::json!([0.0, 1.0])
    );
    assert!(summary["paired_ci_95"].is_null());

    let inspection = Command::new(executable)
        .arg("inspect")
        .arg(&first)
        .args(["2", "--action", "1"])
        .output()
        .unwrap();
    assert!(
        inspection.status.success(),
        "{}",
        String::from_utf8_lossy(&inspection.stderr)
    );
    let trace = String::from_utf8(inspection.stdout).unwrap();
    assert!(trace.contains("Action 1 | Two (random)"));
    assert!(trace.contains("Game 2 | ActionLimit | 1 actions"));
}
