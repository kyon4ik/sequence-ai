//! Champion promotion from checkpoint-bound, paired fixed-panel evidence.
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use anyhow::{Result, ensure};
use clap::Args as ClapArgs;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use sequence_agent::{AgentConfig, AgentDescriptor, AgentSpec};
use sequence_runner::MatchConfig;
use serde::{Deserialize, Serialize};

use crate::artifacts::{digest, read_json, replace_json, write_json};

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    pub incumbent: PathBuf,
    #[arg(long)]
    pub candidate: PathBuf,
    #[arg(long)]
    pub incumbent_spec: PathBuf,
    #[arg(long)]
    pub candidate_spec: PathBuf,
    #[arg(long)]
    pub registry: PathBuf,
    #[arg(long)]
    pub decision: PathBuf,
    #[arg(long, default_value = "0.05")]
    pub family_alpha: f64,
    /// Optional aggregate-study decision containing a boolean `confirmed` field.
    #[arg(long)]
    pub confirmation: Option<PathBuf>,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Protocol {
    games: usize,
    inference_batch_games: usize,
    seed: u64,
}

#[derive(Deserialize)]
struct Report {
    config: Protocol,
    device: String,
    checkpoint: PathBuf,
    checkpoint_sha256: String,
    #[serde(default)]
    opponent_checkpoint_sha256: BTreeMap<String, String>,
    baselines: Vec<Evaluation>,
}

#[derive(Deserialize)]
struct Evaluation {
    baseline: AgentConfig,
    summary: Summary,
    games: Vec<Score>,
}

#[derive(Deserialize)]
struct Summary {
    opponent: String,
    games: usize,
    score: f64,
}

#[derive(Deserialize)]
struct Score {
    config: MatchConfig,
    learner_seat: usize,
    score: f64,
}

#[derive(Debug, PartialEq)]
struct SeedBlock {
    seed: u64,
    agent_seeds: [u64; 2],
    max_actions: [usize; 2],
}

struct Scores {
    schedule: Vec<SeedBlock>,
    opponents: BTreeMap<String, (AgentConfig, Vec<f64>)>,
}

fn load_scores(report: &Report) -> Result<Scores> {
    ensure!(
        report.config.games >= 400 && report.config.games.is_multiple_of(2),
        "gate needs at least 400 paired-seat games per opponent"
    );
    ensure!(
        report.config.inference_batch_games > 0,
        "invalid evaluation batch size"
    );
    crate::parse_device(&report.device).map_err(anyhow::Error::msg)?;
    let mut schedule = None;
    let mut opponents = BTreeMap::new();
    for evaluation in &report.baselines {
        ensure!(
            evaluation.games.len() == report.config.games
                && evaluation.summary.games == report.config.games,
            "evaluation game count mismatch"
        );
        let mut keys = Vec::new();
        let mut scores = Vec::new();
        for [a, b] in evaluation.games.as_chunks::<2>().0 {
            ensure!(
                a.learner_seat == 0
                    && b.learner_seat == 1
                    && a.config.game_seed == b.config.game_seed
                    && a.config.agent_seeds == [b.config.agent_seeds[1], b.config.agent_seeds[0]],
                "invalid seat-swapped seed pairing"
            );
            ensure!(
                [0., 0.5, 1.].contains(&a.score) && [0., 0.5, 1.].contains(&b.score),
                "invalid evaluation score"
            );
            keys.push(SeedBlock {
                seed: a.config.game_seed,
                agent_seeds: a.config.agent_seeds,
                max_actions: [a.config.max_actions, b.config.max_actions],
            });
            scores.push((a.score + b.score) / 2.);
        }
        ensure!(
            keys.iter()
                .map(|key| key.seed)
                .collect::<BTreeSet<_>>()
                .len()
                == keys.len(),
            "duplicate game seeds"
        );
        if let Some(schedule) = &schedule {
            ensure!(schedule == &keys, "opponents have different seed schedules");
        } else {
            schedule = Some(keys);
        }
        ensure!(
            (mean(&scores) - evaluation.summary.score).abs() < 1e-12,
            "score summary mismatch"
        );
        ensure!(
            !evaluation.summary.opponent.is_empty()
                && !opponents.contains_key(&evaluation.summary.opponent),
            "empty or duplicate opponent"
        );
        opponents.insert(
            evaluation.summary.opponent.clone(),
            (evaluation.baseline.clone(), scores),
        );
    }
    ensure!(!opponents.is_empty(), "empty opponent panel");
    Ok(Scores {
        schedule: schedule.unwrap(),
        opponents,
    })
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn lower_bound(values: &[f64], alpha: f64) -> f64 {
    const RESAMPLES: usize = 5000;
    if values.iter().all(|v| *v == values[0]) {
        return values[0];
    }
    let mut rng = StdRng::seed_from_u64(2026111703);
    let mut draws: Vec<_> = (0..RESAMPLES)
        .map(|_| {
            (0..values.len())
                .map(|_| values[rng.random_range(0..values.len())])
                .sum::<f64>()
                / values.len() as f64
        })
        .collect();
    draws.sort_by(f64::total_cmp);
    draws[((alpha * RESAMPLES as f64) as usize).saturating_sub(1)]
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Comparison {
    difference: f64,
    lower_bound: f64,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Gate {
    passed: bool,
    primary_difference: f64,
    primary_lower_bound: f64,
    opponents: BTreeMap<String, Comparison>,
    seed_blocks: usize,
    one_sided_alpha_per_check: f64,
    protocol: String,
}

fn compare(incumbent: &Report, candidate: &Report, family_alpha: f64) -> Result<Gate> {
    ensure!(
        family_alpha.is_finite() && family_alpha > 0. && family_alpha < 1.,
        "invalid family alpha"
    );
    ensure!(
        incumbent.config == candidate.config && incumbent.device == candidate.device,
        "evaluation protocol differs"
    );
    ensure!(
        incumbent.opponent_checkpoint_sha256 == candidate.opponent_checkpoint_sha256,
        "opponent checkpoint hashes differ"
    );
    let old = load_scores(incumbent)?;
    let new = load_scores(candidate)?;
    ensure!(
        old.schedule == new.schedule && old.opponents.keys().eq(new.opponents.keys()),
        "candidate/incumbent seed schedules or panels differ"
    );
    let alpha = family_alpha / (old.opponents.len() + 1) as f64;
    let mut primary = vec![0.; old.schedule.len()];
    let mut opponents = BTreeMap::new();
    for (name, (policy, values)) in &old.opponents {
        let (new_policy, new_values) = &new.opponents[name];
        ensure!(policy == new_policy, "opponent policy differs");
        let differences: Vec<_> = values.iter().zip(new_values).map(|(a, b)| b - a).collect();
        for (total, difference) in primary.iter_mut().zip(&differences) {
            *total += difference / old.opponents.len() as f64;
        }
        opponents.insert(
            name.clone(),
            Comparison {
                difference: mean(&differences),
                lower_bound: lower_bound(&differences, alpha),
            },
        );
    }
    let primary_lower_bound = lower_bound(&primary, alpha);
    Ok(Gate {
        passed: primary_lower_bound > 0. && opponents.values().all(|c| c.lower_bound > -0.05),
        primary_difference: mean(&primary), primary_lower_bound, opponents,
        seed_blocks: old.schedule.len(), one_sided_alpha_per_check: alpha,
        protocol: "Paired seed-block percentile bootstrap; 5000 resamples; StdRng seed 2026111703; Bonferroni across primary plus opponents; strict primary superiority; per-opponent non-regression margin 5 points.".into(),
    })
}

fn verify_opponents(report: &Report) -> Result<()> {
    for evaluation in &report.baselines {
        if let AgentConfig::MonteCarlo { checkpoint } = &evaluation.baseline {
            ensure!(
                report
                    .opponent_checkpoint_sha256
                    .get(&evaluation.summary.opponent)
                    == Some(&digest(checkpoint)?),
                "opponent checkpoint is unbound or changed"
            );
        }
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
struct Binding {
    agent_spec: PathBuf,
    agent_spec_sha256: String,
    checkpoint: PathBuf,
    checkpoint_sha256: String,
}

fn binding(report: &Report, path: &Path) -> Result<Binding> {
    let agent_spec = std::fs::canonicalize(path)?;
    let spec: AgentSpec = read_json(&agent_spec)?;
    let AgentConfig::MonteCarlo { checkpoint } = spec.config else {
        anyhow::bail!("gate requires a neural checkpoint spec");
    };
    let model = std::fs::canonicalize(agent_spec.parent().unwrap().join(checkpoint))?;
    ensure!(
        model == std::fs::canonicalize(&report.checkpoint)?,
        "agent spec and evaluation checkpoint differ"
    );
    ensure!(
        digest(&model)? == report.checkpoint_sha256,
        "evaluation is missing a matching checkpoint hash"
    );
    Ok(Binding {
        agent_spec_sha256: digest(&agent_spec)?,
        agent_spec,
        checkpoint: model,
        checkpoint_sha256: report.checkpoint_sha256.clone(),
    })
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct FileEvidence {
    path: PathBuf,
    sha256: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Inputs {
    incumbent: Binding,
    candidate: Binding,
    incumbent_report: FileEvidence,
    candidate_report: FileEvidence,
    confirmation: Option<FileEvidence>,
    family_alpha: f64,
}

#[derive(Serialize, Deserialize)]
struct Decision {
    #[serde(flatten)]
    gate: Gate,
    aggregate_confirmation_passed: bool,
    inputs: Inputs,
}

#[derive(Serialize, Deserialize)]
struct Registry {
    #[serde(flatten)]
    binding: Binding,
    #[serde(flatten)]
    spec: AgentSpec,
    decision: PathBuf,
}

fn evidence(path: &Path) -> Result<FileEvidence> {
    Ok(FileEvidence {
        path: std::fs::canonicalize(path)?,
        sha256: digest(path)?,
    })
}

fn publish_registry(binding: &Binding, decision: &Path, registry: &Path) -> Result<()> {
    let mut spec: AgentSpec = read_json(&binding.agent_spec)?;
    let source_id = spec.descriptor.id;
    spec.descriptor = AgentDescriptor::new("champion", "Sequence champion");
    spec.descriptor
        .metadata
        .insert("source_agent_id".into(), source_id);
    spec.config = AgentConfig::MonteCarlo {
        checkpoint: binding.checkpoint.clone(),
    };
    replace_json(
        registry,
        &Registry {
            binding: binding.clone(),
            spec,
            decision: std::path::absolute(decision)?,
        },
    )
}

pub fn run(args: &Args) -> Result<()> {
    ensure!(
        args.family_alpha.is_finite() && args.family_alpha > 0. && args.family_alpha < 1.,
        "invalid family alpha"
    );
    let incumbent: Report = read_json(&args.incumbent)?;
    let candidate: Report = read_json(&args.candidate)?;
    ensure!(
        std::path::absolute(&args.registry)? != std::path::absolute(&args.decision)?,
        "registry and decision paths must differ"
    );
    verify_opponents(&incumbent)?;
    verify_opponents(&candidate)?;
    let inputs = Inputs {
        incumbent: binding(&incumbent, &args.incumbent_spec)?,
        candidate: binding(&candidate, &args.candidate_spec)?,
        incumbent_report: evidence(&args.incumbent)?,
        candidate_report: evidence(&args.candidate)?,
        confirmation: args.confirmation.as_deref().map(evidence).transpose()?,
        family_alpha: args.family_alpha,
    };
    #[derive(Deserialize)]
    struct Confirmation {
        confirmed: bool,
    }
    let eligible = args
        .confirmation
        .as_deref()
        .map(read_json::<Confirmation>)
        .transpose()?
        .is_none_or(|c| c.confirmed);
    let prior = if args.decision.exists() {
        let decision: Decision = read_json(&args.decision)?;
        ensure!(
            decision.inputs == inputs && decision.aggregate_confirmation_passed == eligible,
            "immutable gate decision inputs changed"
        );
        Some(decision)
    } else {
        None
    };
    let registered = if args.registry.exists() {
        let registry: Registry = read_json(&args.registry)?;
        let expected = if prior.as_ref().is_some_and(|d| d.gate.passed)
            && registry.binding == inputs.candidate
        {
            &inputs.candidate
        } else {
            &inputs.incumbent
        };
        ensure!(
            &registry.binding == expected,
            "evaluated incumbent is not the registered champion or its spec changed"
        );
        let AgentConfig::MonteCarlo { checkpoint } = &registry.spec.config else {
            anyhow::bail!("champion is not a neural checkpoint");
        };
        ensure!(
            std::fs::canonicalize(checkpoint)? == expected.checkpoint,
            "registered champion checkpoint differs"
        );
        Some(registry)
    } else {
        None
    };
    let decision = if let Some(prior) = prior {
        prior
    } else {
        let mut gate = compare(&incumbent, &candidate, args.family_alpha)?;
        gate.passed &= eligible;
        let decision = Decision {
            gate,
            aggregate_confirmation_passed: eligible,
            inputs,
        };
        if let Some(parent) = args.decision.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        write_json(&args.decision, &decision)?;
        decision
    };
    if decision.gate.passed {
        if registered
            .as_ref()
            .is_none_or(|r| r.binding != decision.inputs.candidate)
        {
            publish_registry(&decision.inputs.candidate, &args.decision, &args.registry)?;
        }
    } else if registered.is_none() {
        publish_registry(&decision.inputs.incumbent, &args.decision, &args.registry)?;
    }
    println!("{}", serde_json::to_string_pretty(&decision)?);
    ensure!(
        decision.gate.passed,
        "promotion gate did not pass; decision saved to {}",
        args.decision.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn report(score: f64) -> Report {
        let games: Vec<_> = (0..200).flat_map(|seed| [
            json!({"config":{"game_seed":seed,"agent_seeds":[1,2],"max_actions":1000},"learner_seat":0,"score":score}),
            json!({"config":{"game_seed":seed,"agent_seeds":[2,1],"max_actions":1000},"learner_seat":1,"score":score}),
        ]).collect();
        serde_json::from_value(
            json!({"checkpoint":"unused.bpk","checkpoint_sha256":"unused","device":"cpu",
            "config":{"games":400,"seed":7,"inference_batch_games":128},
            "baselines":[{"baseline":{"kind":"greedy","weights":[0,0,1,5,10,100]},
            "summary":{"opponent":"greedy","games":400,"score":score},"games":games}]}),
        )
        .unwrap()
    }

    #[test]
    fn gate_requires_gain_and_rejects_protocol_pairing_and_summary_errors() {
        assert!(!compare(&report(0.5), &report(0.5), 0.05).unwrap().passed);
        assert!(compare(&report(0.5), &report(1.), 0.05).unwrap().passed);
        for mutation in 0..6 {
            let mut candidate = report(1.);
            match mutation {
                0 => candidate.config.seed += 1,
                1 => candidate.baselines[0].games[0].learner_seat = 1,
                2 => candidate.baselines[0].games[2].config.game_seed = 0,
                3 => candidate.baselines[0].games[0].score = 0.7,
                4 => candidate.baselines[0].summary.score = 0.,
                _ => {
                    candidate.baselines[0].games[2].config.game_seed = 0;
                    candidate.baselines[0].games[3].config.game_seed = 0;
                }
            }
            assert!(compare(&report(0.5), &candidate, 0.05).is_err());
        }
        for alpha in [0., 1., -0.1, f64::NAN] {
            assert!(compare(&report(0.5), &report(1.), alpha).is_err());
        }
    }

    #[test]
    fn primary_gain_cannot_hide_opponent_regression() {
        let mut old = report(0.5);
        let mut new = report(0.5);
        old.baselines.clear();
        new.baselines.clear();
        for (i, score) in [1., 1., 0.].into_iter().enumerate() {
            let mut a = report(0.5).baselines.remove(0);
            let mut b = report(score).baselines.remove(0);
            a.summary.opponent = i.to_string();
            b.summary.opponent = i.to_string();
            old.baselines.push(a);
            new.baselines.push(b);
        }
        let result = compare(&old, &new, 0.05).unwrap();
        assert!(result.primary_lower_bound > 0.);
        assert!(!result.passed);
    }

    #[test]
    fn bootstrap_is_deterministic_and_keeps_paired_seed_uncertainty() {
        let values: Vec<_> = (0..200)
            .map(|i| if i % 2 == 0 { 1. } else { -1. })
            .collect();
        let lower = lower_bound(&values, 0.025);
        assert!((-0.25..0.).contains(&lower));
        assert_eq!(lower, lower_bound(&values, 0.025));
    }
}
