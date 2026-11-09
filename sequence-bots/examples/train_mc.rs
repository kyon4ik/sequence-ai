use burn::prelude::*;
use burn::tensor::{DeviceFilter, DeviceType};
use sequence_bots::mc::{TrainingConfig, ValueNet, train};

fn main() {
    let mut config = TrainingConfig::default();

    let args: Vec<_> = std::env::args().collect();
    if let Some(value) = args.get(1) {
        config.iterations = value.parse().expect("iterations");
    }
    if let Some(value) = args.get(2) {
        config.games_per_iteration = value.parse().expect("games per iteration");
    }
    let path = args.get(3).map(String::as_str).unwrap_or("mc-model");

    let devices = Device::enumerate(
        DeviceFilter::new()
            .with(DeviceType::Cuda)
            .with(DeviceType::Flex),
    );
    let device = devices.first().unwrap();
    device.seed(config.seed);

    let net = ValueNet::new(device);
    println!("Model has {} parameters", net.num_params());
    let net = train(net, device.clone(), &config);
    net.save_file(path).expect("save model");
    println!("Saved model to {path}");
}
