"""Exercise actual forward hooks with small real PyTorch networks."""
from collections import OrderedDict

import pandas as pd
import pytest
import torch
from torch import nn

from notebook_support import SAVED_TRIALS, load_function


@pytest.fixture(params=SAVED_TRIALS, ids=lambda n: f'trial-{n}-saved')
def collect(request):
    return load_function(request.param, 'collect_layer_activations', {'torch': torch, 'pd': pd})


def test_known_layer_shapes_and_counts(collect):
    """A known network returns exact leaf shapes and per-image activation counts."""
    network = nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(3, 2, 1)), ('relu', nn.ReLU()),
        ('flatten', nn.Flatten()), ('head', nn.Linear(32, 5)),
    ]))
    table = collect(network, image_size=4)
    assert list(table['Layer']) == ['conv', 'relu', 'flatten', 'head']
    assert list(table['Output Shape']) == [(1, 2, 4, 4), (1, 2, 4, 4), (1, 32), (1, 5)]
    assert list(table['Activations']) == [32, 32, 32, 5]
    assert table['Activations'].sum() == 101


@pytest.mark.parametrize('trial', [2, 4], ids=['trial-2-saved', 'trial-4-saved'])
def test_profile_metadata_identifies_model_and_resolution(trial):
    """The two cache-aware profiling variants tag activation tables with model identity and image size."""
    collect = load_function(trial, 'collect_layer_activations', {'torch': torch, 'pd': pd})
    network = nn.Sequential(nn.Conv2d(3, 2, 1))
    table = collect(network, image_size=4)
    assert table.attrs == {'image_size': 4, 'profiled_model_id': id(network)}


def test_preserves_float64_and_mixed_training_modes(collect):
    """Profiling uses the network dtype and restores every module's original training mode."""
    network = nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2)).double()
    network.train()
    network[1].eval()
    modes = [module.training for module in network.modules()]
    before_mean = network[1].running_mean.clone()
    table = collect(network, image_size=4)
    assert list(table['Activations']) == [32, 32]
    assert next(network.parameters()).dtype == torch.float64
    assert [module.training for module in network.modules()] == modes
    assert torch.equal(network[1].running_mean, before_mean)
    assert all(parameter.grad is None for parameter in network.parameters())


def test_failure_removes_new_hooks_and_preserves_existing_hooks(collect):
    """A failing forward pass propagates its error while restoring modes and removing added hooks."""
    class Explode(nn.Module):
        def forward(self, value):
            raise RuntimeError('controlled forward failure')
    network = nn.Sequential(nn.Conv2d(3, 2, 1), Explode())
    network.train()
    network[1].eval()
    existing = network[0].register_forward_hook(lambda *_: None)
    modes = [module.training for module in network.modules()]
    hooks = [set(module._forward_hooks) for module in network.modules()]
    try:
        with pytest.raises(RuntimeError, match='controlled forward failure'):
            collect(network, image_size=4)
        assert [module.training for module in network.modules()] == modes
        assert [set(module._forward_hooks) for module in network.modules()] == hooks
    finally:
        existing.remove()


def test_skips_non_tensor_leaf_outputs(collect):
    """Dictionary outputs are skipped without losing records for earlier tensor-producing leaves."""
    class AsDictionary(nn.Module):
        def forward(self, value):
            return {'prediction': value}
    network = nn.Sequential(nn.Conv2d(3, 2, 1), AsDictionary())
    table = collect(network, image_size=4)
    assert list(table['Layer']) == ['0']
    assert list(table['Activations']) == [32]


def test_records_repeated_invocations_of_shared_leaf(collect):
    """Reusing a leaf twice records both invocations instead of silently deduplicating activations."""
    class ReusedLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Conv2d(3, 3, 1)
        def forward(self, value):
            return self.shared(self.shared(value))
    table = collect(ReusedLayer(), image_size=4)
    assert list(table['Layer']) == ['shared', 'shared']
    assert list(table['Activations']) == [48, 48]


def test_forward_disables_gradients_and_retains_existing_hook(collect):
    """A successful profile disables gradient tracking, preserves a pre-existing hook and cleans up its own hooks."""
    network = nn.Sequential(nn.Conv2d(3, 2, 1), nn.ReLU())
    enabled = []
    handle = network[0].register_forward_hook(lambda *_: enabled.append(torch.is_grad_enabled()))
    hooks = [set(module._forward_hooks) for module in network.modules()]
    try:
        collect(network, image_size=4)
        assert enabled == [False]
        assert [set(module._forward_hooks) for module in network.modules()] == hooks
    finally:
        handle.remove()


@pytest.mark.parametrize('trial', SAVED_TRIALS, ids=lambda n: f'trial-{n}-original')
def test_original_hook_counts_one_image_not_the_whole_batch(trial):
    """The original training notebook hook counts one image from a two-image tensor batch."""
    records = []
    make_hook = load_function(trial, 'make_hook', {'torch': torch, 'layer_info': records}, saved=False)
    make_hook('example')(nn.Identity(), (), torch.zeros(2, 3, 4, 4))
    assert records == [{'Layer': 'example', 'Type': 'Identity', 'Output Shape': (2, 3, 4, 4), 'Activations': 48}]


@pytest.mark.parametrize('trial', SAVED_TRIALS, ids=lambda n: f'trial-{n}-original')
def test_original_hook_ignores_non_tensor_output(trial):
    """The original hook safely ignores a dictionary output rather than counting or crashing on it."""
    records = []
    make_hook = load_function(trial, 'make_hook', {'torch': torch, 'layer_info': records}, saved=False)
    make_hook('example')(nn.Identity(), (), {'prediction': torch.zeros(1, 3)})
    assert records == []

