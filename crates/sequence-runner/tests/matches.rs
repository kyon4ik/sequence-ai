use sequence_agent::{
    Agent, AgentConfig, AgentDescriptor, AgentError, AgentFactory, AgentSpec, EpisodeContext,
    EpisodeResult, EpisodeStatus, RandomAgent,
};
use sequence_core::{
    Action, Classic, ClassicObservation, Color, Game, GameOutcome, Simple, SimpleObservation,
};
use sequence_runner::{
    EventSink, MatchConfig, MatchEvent, MatchRecord, MatchRunner, Participant, RecordError,
    RecordedGame, Replay, ReplayError, Reward, Termination,
};

fn spec(id: &str, config: AgentConfig) -> AgentSpec {
    AgentSpec {
        descriptor: AgentDescriptor::new(id, id),
        config,
    }
}

#[test]
fn complete_matches_record_events_rewards_and_replay_after_serialization() {
    let random = spec("random", AgentConfig::Random);
    let greedy = spec("greedy", AgentConfig::greedy());
    for seed in 0..8 {
        let mut events = Vec::new();
        let record = MatchRunner::<Simple>::default()
            .run_factories_with_sink(
                MatchConfig {
                    game_seed: seed,
                    ..Default::default()
                },
                [&random, &greedy],
                &mut events,
            )
            .unwrap();
        let outcome = record.termination.outcome().unwrap();
        assert_ne!(outcome, GameOutcome::InProgress);
        assert!(record.metrics.turns <= 96);
        assert_eq!(record.metrics.actions, record.steps.len());
        assert_eq!(
            record
                .metrics
                .players
                .iter()
                .map(|p| p.decisions)
                .sum::<usize>(),
            record.steps.len()
        );
        assert_eq!(events.len(), record.steps.len() + 2);
        assert_eq!(events[0], MatchEvent::Started(record.header.clone()));
        assert_eq!(
            events.last().unwrap(),
            &MatchEvent::Finished {
                termination: record.termination.clone(),
                metrics: record.metrics.clone(),
            }
        );
        for (step, event) in record.steps.iter().zip(&events[1..events.len() - 1]) {
            assert_eq!(event, &MatchEvent::Action(step.clone()));
        }
        for (index, player) in record.metrics.players.iter().enumerate() {
            let expected_reward = match outcome {
                GameOutcome::Won(winner) if winner as usize == index => 1.,
                GameOutcome::Won(_) => -1.,
                _ => 0.,
            };
            assert_eq!(player.reward, expected_reward);
            assert_eq!(
                player.decision_time,
                record
                    .steps
                    .iter()
                    .filter(|s| s.actor as usize == index)
                    .map(|s| s.decision_time)
                    .sum()
            );
            assert!(player.max_decision_time <= player.decision_time);
        }
        let json = serde_json::to_string(&record).unwrap();
        let loaded: MatchRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(record, loaded);
        let mut replay = Replay::<Simple>::new(&loaded).unwrap();
        assert_eq!(replay.frame_count(), loaded.frame_count());
        for index in (0..loaded.frame_count()).rev() {
            assert_eq!(
                &replay.seek(index).unwrap().snapshot(),
                loaded.frame(index).unwrap()
            );
        }
        assert!(replay.step_back().is_none());
        replay.step_forward().unwrap();
        assert_eq!(replay.position(), 1);
        replay.step_back().unwrap();
        assert_eq!(replay.position(), 0);
        let last = loaded.frame_count() - 1;
        replay.seek(last).unwrap();
        assert!(replay.step_forward().is_none());
        assert!(replay.seek(last + 1).is_err());
        assert_eq!(replay.position(), last);
    }
}

#[test]
fn independent_seeds_reproduce_actions_with_fresh_or_reused_agents() {
    let mut runner = MatchRunner::<Simple>::default();
    let mut red = RandomAgent::from_seed(900);
    let mut blue = RandomAgent::from_seed(901);
    let mut play = || {
        runner
            .run(
                MatchConfig::default(),
                [
                    Participant::new(AgentDescriptor::new("red", "R"), &mut red),
                    Participant::new(AgentDescriptor::new("blue", "B"), &mut blue),
                ],
            )
            .unwrap()
    };
    let first = play();
    let second = play();
    assert_eq!(first.termination, second.termination);
    assert_eq!(
        first
            .steps
            .iter()
            .map(|s| (s.action, &s.state))
            .collect::<Vec<_>>(),
        second
            .steps
            .iter()
            .map(|s| (s.action, &s.state))
            .collect::<Vec<_>>()
    );
    let factory = spec("random", AgentConfig::Random);
    let fresh = runner
        .run_factories(MatchConfig::default(), [&factory, &factory])
        .unwrap();
    assert_eq!(
        first.steps.iter().map(|s| s.action).collect::<Vec<_>>(),
        fresh.steps.iter().map(|s| s.action).collect::<Vec<_>>()
    );
}

#[derive(Default)]
struct Hooks {
    starts: Vec<EpisodeContext>,
    ends: Vec<EpisodeResult>,
}

impl Agent for Hooks {
    fn choose_action(
        &mut self,
        _: SimpleObservation<'_>,
        actions: &[Action],
    ) -> Result<Action, AgentError> {
        Ok(actions[0])
    }
    fn on_episode_start(&mut self, context: EpisodeContext) {
        self.starts.push(context);
    }
    fn on_episode_end(&mut self, result: EpisodeResult) {
        self.ends.push(result);
    }
}

#[test]
fn truncation_is_distinct_from_draw_and_notifies_both_agents() {
    let mut red = Hooks::default();
    let mut blue = Hooks::default();
    let record = MatchRunner::<Simple>::default()
        .run(
            MatchConfig {
                max_actions: 1,
                ..Default::default()
            },
            [
                Participant::new(AgentDescriptor::new("red", "R"), &mut red),
                Participant::new(AgentDescriptor::new("blue", "B"), &mut blue),
            ],
        )
        .unwrap();
    assert_eq!(record.termination, Termination::ActionLimit);
    assert_eq!(record.termination.outcome(), None);
    assert_eq!(record.metrics.players[0].reward, 0.);
    assert_eq!(blue.starts[0].color, Color::Blue);
    assert_eq!(blue.starts[0].agent_seed, 1);
    for result in [red.ends[0], blue.ends[0]] {
        assert!(matches!(result.status, EpisodeStatus::Truncated));
        assert_eq!(result.outcome, GameOutcome::InProgress);
        assert_eq!(result.reward, 0.);
    }
    assert_eq!(blue.ends[0].actions, 0);
    Replay::<Simple>::new(&record).unwrap();
}

struct Broken(bool);
impl Agent for Broken {
    fn choose_action(
        &mut self,
        _: SimpleObservation<'_>,
        _: &[Action],
    ) -> Result<Action, AgentError> {
        if self.0 {
            Err("inference failed".into())
        } else {
            Ok(Action::Pass)
        }
    }
}

#[test]
fn inference_and_illegal_action_errors_keep_partial_replays_and_emit_abort() {
    for inference_error in [true, false] {
        let mut broken = Broken(inference_error);
        let mut opponent = Hooks::default();
        let mut events = Vec::new();
        let error = MatchRunner::<Simple>::default()
            .run_with_sink(
                MatchConfig::default(),
                [
                    Participant::new(AgentDescriptor::new("bad", "Bad"), &mut broken),
                    Participant::new(AgentDescriptor::new("good", "Good"), &mut opponent),
                ],
                &mut events,
            )
            .unwrap_err();
        let record = error.partial_record().unwrap();
        assert!(matches!(record.termination, Termination::Aborted { .. }));
        assert_eq!(record.metrics.players[0].decisions, 1);
        assert!(record.steps.is_empty());
        assert!(matches!(
            events.last(),
            Some(MatchEvent::Finished {
                termination: Termination::Aborted { .. },
                ..
            })
        ));
        assert!(matches!(opponent.ends[0].status, EpisodeStatus::Aborted));
        Replay::<Simple>::new(record).unwrap();
    }
}

struct FailingSink {
    calls: usize,
    fail_at: usize,
}
impl EventSink for FailingSink {
    fn record(&mut self, _: &MatchEvent) -> Result<(), RecordError> {
        self.calls += 1;
        if self.calls == self.fail_at {
            Err("storage full".into())
        } else {
            Ok(())
        }
    }
}

#[test]
fn recorder_failures_at_start_action_and_finish_are_reported_without_losing_moves() {
    let random = spec("random", AgentConfig::Random);
    for (fail_at, steps) in [(1, 0), (2, 1), (3, 1)] {
        let mut sink = FailingSink { calls: 0, fail_at };
        let error = MatchRunner::<Simple>::default()
            .run_factories_with_sink(
                MatchConfig {
                    max_actions: 1,
                    ..Default::default()
                },
                [&random, &random],
                &mut sink,
            )
            .unwrap_err();
        let record = error.partial_record().unwrap();
        assert_eq!(sink.calls, fail_at);
        assert_eq!(record.steps.len(), steps);
        assert!(error.to_string().contains("storage full"));
        Replay::<Simple>::new(record).unwrap();
    }
}

struct ShapedReward {
    moves: usize,
    invalid: bool,
}
impl Reward<Simple> for ShapedReward {
    fn id(&self) -> &str {
        "test-shaping"
    }
    fn reset(&mut self) {
        self.moves = 0;
    }
    fn rewards(&mut self, before: &Simple, _: Action, after: &Simple) -> [f64; 2] {
        assert_eq!(after.turns(), before.turns() + 1);
        self.moves += 1;
        if self.invalid {
            [f64::NAN, 0.]
        } else {
            [self.moves as f64, -0.5]
        }
    }
}

#[test]
fn rewards_reset_each_episode_and_count_both_seats() {
    let random = spec("random", AgentConfig::Random);
    let mut runner = MatchRunner::with_reward(ShapedReward {
        moves: 50,
        invalid: false,
    });
    for _ in 0..2 {
        let record = runner
            .run_factories(
                MatchConfig {
                    max_actions: 3,
                    ..Default::default()
                },
                [&random, &random],
            )
            .unwrap();
        assert_eq!(record.header.reward_id, "test-shaping");
        assert_eq!(record.metrics.players[0].reward, 6.);
        assert_eq!(record.metrics.players[1].reward, -1.5);
        Replay::<Simple>::new(&record).unwrap();
    }
    let mut events = Vec::new();
    let error = MatchRunner::with_reward(ShapedReward {
        moves: 0,
        invalid: true,
    })
    .run_factories_with_sink(MatchConfig::default(), [&random, &random], &mut events)
    .unwrap_err();
    let record = error.partial_record().unwrap();
    assert_eq!(record.steps.len(), 1);
    assert_eq!(events[1], MatchEvent::Action(record.steps[0].clone()));
    assert!(serde_json::to_string(record).is_ok());
    Replay::<Simple>::new(record).unwrap();
}

struct ExchangeFirst;
impl Agent<Classic> for ExchangeFirst {
    fn choose_action(
        &mut self,
        _: ClassicObservation<'_>,
        actions: &[Action],
    ) -> Result<Action, AgentError> {
        Ok(actions
            .iter()
            .find(|a| matches!(a, Action::Exchange { .. }))
            .copied()
            .unwrap_or(actions[0]))
    }
}

#[test]
fn classic_replays_preserve_same_actor_exchanges_and_turn_counts() {
    let mut exchanges = 0;
    for seed in 0..16 {
        let mut red = ExchangeFirst;
        let mut blue = RandomAgent::from_seed(7);
        let record = MatchRunner::<Classic>::default()
            .run(
                MatchConfig {
                    game_seed: seed,
                    agent_seeds: [1, 7],
                    ..Default::default()
                },
                [
                    Participant::new(AgentDescriptor::new("exchange", "Exchange"), &mut red),
                    Participant::new(AgentDescriptor::new("random", "Random"), &mut blue),
                ],
            )
            .unwrap();
        assert!(record.termination.outcome().is_some());
        let mut replay = Replay::<Classic>::new(&record).unwrap();
        for step in &record.steps {
            let before = record.frame(step.index).unwrap();
            if matches!(step.action, Action::Exchange { .. }) {
                exchanges += 1;
                assert_eq!(step.state.turns, before.turns);
                assert_eq!(step.state.current_player, before.current_player);
                assert!(step.state.classic.as_ref().unwrap().exchanged_dead_card);
            }
            assert_eq!(
                &replay.seek(step.index + 1).unwrap().snapshot(),
                &step.state
            );
        }
    }
    assert!(exchanges > 0);
}

#[test]
fn replay_rejects_incompatible_formats_wrong_games_corrupt_actions_and_metrics() {
    let random = spec("random", AgentConfig::Random);
    let record = MatchRunner::<Simple>::default()
        .run_factories(
            MatchConfig {
                max_actions: 2,
                ..Default::default()
            },
            [&random, &random],
        )
        .unwrap();
    assert!(matches!(
        Replay::<Classic>::new(&record),
        Err(ReplayError::Game { .. })
    ));
    let mut bad = record.clone();
    bad.header.format_version += 1;
    assert!(matches!(
        Replay::<Simple>::new(&bad),
        Err(ReplayError::Format(_))
    ));
    let mut bad = record.clone();
    bad.header.config.game_seed += 1;
    assert!(Replay::<Simple>::new(&bad).is_err());
    let mut bad = record.clone();
    bad.steps[0].action = Action::Pass;
    assert!(Replay::<Simple>::new(&bad).is_err());
    let mut bad = record.clone();
    bad.steps[0].actor = Color::Blue;
    assert!(Replay::<Simple>::new(&bad).is_err());
    let mut bad = record.clone();
    bad.steps[0].state.hand.clear();
    assert!(Replay::<Simple>::new(&bad).is_err());
    let mut bad = record.clone();
    bad.metrics.players[0].reward = 99.;
    assert!(Replay::<Simple>::new(&bad).is_err());
    let mut bad = record.clone();
    bad.termination = Termination::Completed(GameOutcome::Draw);
    assert!(Replay::<Simple>::new(&bad).is_err());
    let mut bad = record.clone();
    bad.header.config.max_actions = 3;
    assert!(Replay::<Simple>::new(&bad).is_err());
    assert!(serde_json::from_str::<sequence_core::Position>("100").is_err());
    assert!(serde_json::from_str::<sequence_core::Card>("52").is_err());
}

#[test]
fn invalid_setup_and_factory_errors_are_fallible_before_execution() {
    let random = spec("random", AgentConfig::Random);
    let mut runner = MatchRunner::<Simple>::default();
    let error = runner
        .run_factories(
            MatchConfig {
                max_actions: 0,
                ..Default::default()
            },
            [&random, &random],
        )
        .unwrap_err();
    assert!(error.partial_record().is_none());
    let empty = spec(" ", AgentConfig::Random);
    assert!(
        runner
            .run_factories(MatchConfig::default(), [&empty, &random])
            .is_err()
    );
    struct BadFactory;
    impl AgentFactory for BadFactory {
        fn descriptor(&self) -> AgentDescriptor {
            AgentDescriptor::new("bad", "Bad")
        }
        fn create(&self, _: u64) -> Result<Box<dyn Agent>, AgentError> {
            Err("missing model".into())
        }
    }
    let error = runner
        .run_factories(MatchConfig::default(), [&BadFactory, &random])
        .unwrap_err();
    assert!(error.partial_record().is_none());
    assert!(error.to_string().contains("missing model"));
}
