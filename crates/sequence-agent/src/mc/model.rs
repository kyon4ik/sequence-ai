use super::EncodedAfterstate;
use burn::{
    nn::{
        Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::relu,
};

#[derive(Module, Debug)]
pub struct ValueNet {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    hidden: Linear,
    value: Linear,
}

impl ValueNet {
    pub fn new(device: &Device) -> Self {
        let conv = |input| {
            Conv2dConfig::new([input, 32], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        Self {
            conv1: conv(4),
            conv2: conv(32),
            conv3: conv(32),
            conv4: conv(32),
            hidden: LinearConfig::new(3250, 128).init(device),
            value: LinearConfig::new(128, 1).init(device),
        }
    }

    pub fn forward(&self, board: Tensor<4>, context: Tensor<2>) -> Tensor<2> {
        let x = relu(self.conv1.forward(board));
        let x = relu(self.conv2.forward(x));
        let x = relu(self.conv3.forward(x));
        let x = relu(self.conv4.forward(x)).flatten(1, 3);
        let x = Tensor::cat(vec![x, context], 1);
        let x = relu(self.hidden.forward(x));
        self.value.forward(x).tanh()
    }
}

/// Packs encoded afterstates for inference or training.
pub fn batch(items: &[EncodedAfterstate], device: &Device) -> (Tensor<4>, Tensor<2>) {
    assert!(!items.is_empty());
    let boards: Vec<_> = items.iter().flat_map(|x| x.board).collect();
    let contexts: Vec<_> = items.iter().flat_map(|x| x.context).collect();
    (
        Tensor::from_data(TensorData::new(boards, [items.len(), 4, 10, 10]), device),
        Tensor::from_data(TensorData::new(contexts, [items.len(), 50]), device),
    )
}
