use sequence_agent::{AgentConfig, AgentDescriptor, AgentFactory, AgentSpec, InferenceDevice};
use sequence_core::{Game, Simple};

#[test]
fn saved_heuristic_specs_restore_parameters_and_seeded_policies() {
    for config in [
        AgentConfig::Random,
        AgentConfig::greedy(),
        AgentConfig::Greedy { weights: [-1; 6] },
    ] {
        let spec = AgentSpec {
            descriptor: AgentDescriptor::new("test", "Test"),
            config,
        };
        let loaded: AgentSpec =
            serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
        assert_eq!(loaded, spec);
        let mut first = spec.create(42).unwrap();
        let mut second = loaded.create(42).unwrap();
        let mut game = Simple::from_seed(7);
        let mut actions = Vec::new();
        while !game.is_game_over() {
            game.legal_actions(&mut actions);
            let action = first.choose_action(game.observation(), &actions).unwrap();
            assert_eq!(
                action,
                second.choose_action(game.observation(), &actions).unwrap()
            );
            assert!(actions.contains(&action));
            game.apply(action).unwrap();
        }
        assert!(first.choose_action(game.observation(), &[]).is_err());
    }
}

#[test]
fn missing_checkpoint_is_an_error_even_when_cuda_is_requested() {
    let config = AgentConfig::MonteCarlo {
        checkpoint: "/nonexistent/sequence-model.bpk".into(),
    };
    let error = match config.create(0, InferenceDevice::Cuda(0)) {
        Ok(_) => panic!("missing model unexpectedly loaded"),
        Err(error) => error,
    };
    #[cfg(feature = "burn")]
    assert!(matches!(
        error,
        sequence_agent::BuildError::Checkpoint { .. }
    ));
    #[cfg(not(feature = "burn"))]
    assert!(matches!(error, sequence_agent::BuildError::BurnDisabled));
}
