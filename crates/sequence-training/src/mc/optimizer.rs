//! Burn 0.22's ModuleOptimizer loader uses Device::default(), which selects CUDA
//! in CUDA-enabled builds. Materialize Adam records on Flex so CPU resume never
//! needs a GPU. Burn moves each state to its parameter's device before updating.
use burn::{
    optim::{
        Adam, AdamConfig, AdamState, ModuleOptimizer, Optimizer, RecordState, StateSink,
        StateSource,
    },
    prelude::*,
};

#[derive(Clone)]
struct PortableAdam(Adam);

#[derive(Clone)]
struct PortableState<const D: usize>(AdamState<D>);

impl<const D: usize> RecordState for PortableState<D> {
    fn state_flatten(&self, prefix: &str, out: &mut StateSink) {
        self.0.state_flatten(prefix, out);
    }

    fn state_unflatten(prefix: &str, src: &mut StateSource, _: &Device) -> Option<Self> {
        AdamState::state_unflatten(prefix, src, &Device::flex()).map(Self)
    }
}

impl Optimizer for PortableAdam {
    type State<const D: usize> = PortableState<D>;

    fn step<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let (tensor, state) = self.0.step(lr, tensor, grad, state.map(|state| state.0));
        (tensor, state.map(PortableState))
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &Device) -> Self::State<D> {
        PortableState(Adam::to_device(state.0, device))
    }
}

pub(super) fn adam() -> ModuleOptimizer {
    PortableAdam(AdamConfig::new().build()).into()
}
