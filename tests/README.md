# Software tests

This suite tests the actual functions and selected analysis cells in all ten notebooks documented under `docs/source-notebooks/`. It does not duplicate those functions in a separate implementation.

`notebook_support.py` extracts unchanged function definitions with Python's syntax tree tools, so installation cells, Drive mounting and training calls are not run. Matrix and timing tests execute the original analysis cells with small controlled inputs. The roofline tests omit only import statements so a controlled FLOP-profiler result can be supplied.

## What is checked

| Area | Cases | Verified behaviour |
| --- | ---: | --- |
| Activations and hooks | 42 | Known shapes/counts, float64 inputs, training-mode restoration, disabled gradients, hook cleanup on success and failure, shared-module reuse, non-tensor outputs, original per-image batch counting, and metadata in the two cache-aware variants. |
| Evaluation orchestration | 20 | Saved checkpoint selection, validation settings, returned model/metrics, actual output directory, missing plot files, and error propagation. |
| Analysis cells | 16 | Confusion-matrix denominators, zero columns, wrinkle filtering, background retention, saved plots, timing arithmetic, FP32 model-copy profiling, roofline units, and rejection of invalid FLOP estimates. |

There are 78 parameterised cases across the notebook implementations. Repeated cases exercise each supplied trial version; this is not a claim of 78 distinct algorithms or complete code coverage.

## Run the suite

From the repository root, use Python 3.12 or a compatible version. A virtual environment is recommended.

```text
python -m venv .venv
```

Activate it using `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on Linux/macOS. For a CPU-only environment, install PyTorch from its CPU wheel index before the remaining requirements:

```text
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pytest tests --html=reports/test_report.html --self-contained-html --junitxml=reports/junit.xml
```

The final command returns a nonzero exit code if tests fail. It also refreshes `reports/test_summary.json`, recording case outcomes, versions, and SHA-256 hashes of the tested notebook snapshots and test-suite files. The recorded run used Windows, Python 3.12.14 and CPU PyTorch; exact library versions are in the summary.

## Test boundaries

PyTorch, NumPy, pandas, Pillow, Matplotlib and Seaborn are real libraries in these tests. Tiny local neural networks exercise the activation functions, and synthetic matrices exercise numerical and plotting logic. YOLO loading/validation and Ultralytics FLOP profiling use test doubles to avoid downloading checkpoints, accessing Drive or running expensive evaluation.

These tests do not establish training convergence, accuracy on the dataset, checkpoint compatibility with Ultralytics, actual GPU speed, or performance under production-line conditions. The earlier notebooks at the repository root are not the execution targets of this suite; the ten versioned documentation snapshots are.

