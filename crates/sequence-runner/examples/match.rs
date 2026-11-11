use sequence_agent::{AgentConfig, AgentDescriptor, AgentSpec};
use sequence_core::Simple;
use sequence_runner::{MatchConfig, MatchRunner};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let random = AgentSpec {
        descriptor: AgentDescriptor::new("random-v1", "Random"),
        config: AgentConfig::Random,
    };
    let greedy = AgentSpec {
        descriptor: AgentDescriptor::new("greedy-v1", "Greedy"),
        config: AgentConfig::greedy(),
    };
    let record = MatchRunner::<Simple>::default()
        .run_factories(MatchConfig::default(), [&random, &greedy])?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &record)?;
    Ok(())
}
