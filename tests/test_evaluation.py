"""Test evaluation orchestration while keeping external model loading controlled."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from PIL import Image
import pytest

from notebook_support import SAVED_TRIALS, load_function

PLOTS = ['BoxF1_curve.png', 'BoxP_curve.png', 'BoxR_curve.png', 'BoxPR_curve.png']


@pytest.fixture(params=SAVED_TRIALS, ids=lambda n: f'trial-{n}-saved')
def trial(request):
    return request.param


def evaluation_function(trial, factory):
    return load_function(trial, 'evaluate_and_plot', {'YOLO': factory, 'Path': Path, 'plt': plt, 'Image': Image})


@pytest.mark.parametrize('available', [4, 1], ids=['all-curves', 'missing-curves'])
def test_loads_actual_output_curves_and_reports_missing_files(trial, available, tmp_path, monkeypatch):
    """Evaluation reads curves from the returned run directory and visibly reports missing plot files."""
    actual_dir = tmp_path / 'actual-validation-output'
    actual_dir.mkdir()
    for name in PLOTS[:available]:
        Image.new('RGB', (4, 4), color='green').save(actual_dir / name)
    metrics = SimpleNamespace(save_dir=actual_dir)
    calls = []
    class Model:
        def val(self, **kwargs):
            calls.append(kwargs)
            return metrics
    model = Model()
    loaded = []
    def factory(weights):
        loaded.append(weights)
        return model
    monkeypatch.setattr(plt, 'show', lambda: None)
    fn = evaluation_function(trial, factory)
    returned_model, returned_metrics = fn('trained.pt', 'dataset.yaml', str(tmp_path / 'project'), image_size=8, batch_size=2)
    assert loaded == ['trained.pt']
    assert returned_model is model and returned_metrics is metrics
    assert calls[0]['data'] == 'dataset.yaml'
    assert calls[0]['imgsz'] == 8 and calls[0]['batch'] == 2
    assert calls[0]['split'] == 'val' and calls[0]['plots'] is True
    axes = plt.gcf().axes
    assert [axis.get_title() for axis in axes] == ['F1 Curve', 'Precision Curve', 'Recall Curve', 'Precision-Recall Curve']
    assert sum(bool(axis.images) for axis in axes) == available
    assert sum(any('Not found' in text.get_text() for text in axis.texts) for axis in axes) == 4 - available


def test_default_validation_configuration(trial, tmp_path, monkeypatch):
    """Calling evaluation without overrides uses the notebook's 1280-pixel, batch-four defaults."""
    calls = []
    class Model:
        def val(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(save_dir=tmp_path)
    monkeypatch.setattr(plt, 'show', lambda: None)
    evaluation_function(trial, lambda _: Model())('trained.pt', 'dataset.yaml', str(tmp_path))
    assert calls[0]['imgsz'] == 1280
    assert calls[0]['batch'] == 4
    assert calls[0]['project'] == str(tmp_path)


def test_checkpoint_errors_are_not_silenced(trial, tmp_path):
    """A checkpoint-loading failure reaches the caller instead of producing misleading plots or results."""
    def missing_checkpoint(_):
        raise FileNotFoundError('controlled missing checkpoint')
    with pytest.raises(FileNotFoundError, match='controlled missing checkpoint'):
        evaluation_function(trial, missing_checkpoint)('missing.pt', 'dataset.yaml', str(tmp_path))
    assert not plt.get_fignums()

