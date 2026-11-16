#[cfg(feature = "burn")]
fn main() -> Result<(), sequence_agent::AgentError> {
    use sequence_training::mc::ablation::{AblationConfig, run_ablation};
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: ablation <config.json> <new-output-directory>".into());
    }
    let config: AblationConfig = serde_json::from_reader(std::fs::File::open(&args[0])?)?;
    let device = burn::prelude::Device::cuda(0);
    device.seed(config.seed);
    run_ablation(&config, device, std::path::Path::new(&args[1]))?;
    Ok(())
}
#[cfg(not(feature = "burn"))]
fn main() {
    eprintln!("ablation requires --features burn");
    std::process::exit(1);
}
