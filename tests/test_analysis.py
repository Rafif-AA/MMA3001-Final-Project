"""Check the actual matrix, timing and roofline cells using known inputs."""
from copy import deepcopy
from math import isfinite
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn

from notebook_support import SAVED_TRIALS, find_cell, load_function, run_cell


@pytest.mark.parametrize('trial', SAVED_TRIALS, ids=lambda n: f'trial-{n}-saved')
def test_confusion_matrix_preserves_denominators_and_handles_zero_columns(trial, tmp_path, monkeypatch):
    """Filtering wrinkle after normalisation preserves true-class denominators, zero columns and background."""
    raw = np.array([[2, 0, 0, 0, 4, 0], [1, 0, 0, 0, 0, 0],
                    [0, 0, 3, 0, 0, 0], [0, 0, 0, 4, 0, 0],
                    [1, 0, 0, 0, 8, 0], [0, 0, 0, 1, 0, 0]])
    original = raw.copy()
    metrics = SimpleNamespace(confusion_matrix=SimpleNamespace(matrix=raw), save_dir=tmp_path)
    model = SimpleNamespace(names={0: 'loose-meat', 1: 'packaging-error', 2: 'twisted-meat', 3: 'unsealed', 4: 'wrinkle'})
    monkeypatch.setattr(plt, 'show', lambda: None)
    scope = run_cell(trial, 'normalized = np.divide', {'metrics': metrics, 'model': model})
    assert scope['visible_labels'] == ['loose-meat', 'packaging-error', 'twisted-meat', 'unsealed', 'background']
    assert scope['normalized'][0, 0] == pytest.approx(0.5)
    assert scope['filtered'][:, 0].sum() == pytest.approx(0.75)
    assert np.isfinite(scope['normalized']).all()
    assert (scope['normalized'][:, 1] == 0).all()
    assert scope['filtered'].shape == (5, 5)
    assert np.array_equal(raw, original)
    for name in ['confusion_matrix_no_wrinkle.png', 'confusion_matrix_normalized_no_wrinkle.png']:
        with Image.open(tmp_path / name) as image:
            assert image.width > 500 and image.height > 500


@pytest.mark.parametrize('trial', SAVED_TRIALS, ids=lambda n: f'trial-{n}-saved')
def test_timing_separates_inference_from_total_processing(trial, capsys):
    """Known validation speed values produce separate, correct inference and total-processing times."""
    calls = []
    class Model:
        def val(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(speed={'preprocess': 2.0, 'inference': 4.25, 'postprocess': 0.75})
    scope = run_cell(trial, 'processing_ms = sum(', {'model': Model(), 'DATA_PATH': 'dataset.yaml'})
    assert scope['processing_ms'] == pytest.approx(7.0)
    output = capsys.readouterr().out
    assert 'Inference: 4.25 ms/image' in output
    assert 'postprocessing: 7.00 ms/image' in output
    assert calls == [dict(data='dataset.yaml', split='val', imgsz=1280, batch=1,
                         device=0, rect=False, augment=False, plots=False)]


def run_roofline(trial, flop_result):
    """Run the real cell with a controlled profiler, avoiding an Ultralytics install."""
    network = nn.Sequential(nn.Conv2d(3, 1, 1)).double()
    model = SimpleNamespace(model=network)
    collect = load_function(trial, 'collect_layer_activations', {'torch': torch, 'pd': pd})
    profiles = []
    def profiler(profile_network, imgsz):
        profiles.append((profile_network, imgsz))
        return flop_result
    path, number, code = find_cell(trial, 'model_gflops = float(')
    # Skip only imports so the local test profiler replaces the external API.
    # All assignments, guards, calculations and print statements are unchanged.
    import ast
    tree = ast.parse(code)
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    scope = dict(model=model, collect_layer_activations=collect, torch=torch,
                 deepcopy=deepcopy, isfinite=isfinite, get_flops=profiler)
    exec(compile(tree, f'{path.name}:cell-{number}', 'exec'), scope)
    return scope, network, profiles


@pytest.mark.parametrize('trial', [2, 4], ids=['trial-2-saved', 'trial-4-saved'])
def test_roofline_profiles_a_copy_and_uses_consistent_units(trial):
    """The 1280 roofline cell profiles an FP32 copy and calculates a known intensity and ceiling."""
    scope, network, profiles = run_roofline(trial, 2.0)
    assert len(profiles) == 1 and profiles[0][1] == 1280
    assert profiles[0][0] is not network
    assert next(profiles[0][0].parameters()).dtype == torch.float32
    assert next(network.parameters()).dtype == torch.float64
    # One 1x1 convolution has four parameters and 1,638,400 output values.
    assert scope['total_params'] == 4
    assert scope['total_activations'] == 1_638_400
    assert scope['total_memory_accessed_bytes'] == 13_107_216
    assert scope['arithmetic_intensity'] == pytest.approx(2_000_000_000 / 13_107_216)
    assert scope['ridge_point'] == pytest.approx(73.262366938)
    assert scope['estimated_roofline_flops_s'] == 117_000_000_000_000
    assert scope['bound_type'] == 'COMPUTE-LIMITED'


@pytest.mark.parametrize('trial', [2, 4], ids=['trial-2-saved', 'trial-4-saved'])
@pytest.mark.parametrize('invalid', [0.0, float('nan')], ids=['zero-flops', 'nan-flops'])
def test_roofline_rejects_invalid_profiler_results(trial, invalid):
    """Zero or non-finite profiler results raise a clear error instead of a plausible-looking roofline."""
    with pytest.raises(RuntimeError, match='FLOP profiling failed'):
        run_roofline(trial, invalid)
