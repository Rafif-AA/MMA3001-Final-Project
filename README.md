# MMA3001 Final Project: Pork Rasher Packaging Defect Detection

This project investigates automated inspection of pork rasher packaging using Ultralytics YOLO26 object detection. It compares five training configurations to assess how model size, training duration, and image resolution affect defect detection and inference time.

The engineering objective is to identify defective packages while balancing missed defects, false alarms, and inspection speed. The work was developed in Google Colab for the MMA3001 final project.

**Selected configuration:** Trial 2 - YOLO26 Nano, 100 training epochs, and 1280 x 1280 input images. The final report selects this configuration for its balance of detection performance, class recall, and reported inference time.

## Branches

This repository contains three branches:

| Branch | Purpose |
| --- | --- |
| [main](https://github.com/Rafif-AA/MMA3001-Final-Project/tree/main) | Main project branch containing the Google Colab training notebooks, trial configurations, and project documentation. |
| [PyCharm-Migration-1](https://github.com/Rafif-AA/MMA3001-Final-Project/tree/PyCharm-Migration-1) | Earlier work towards moving development from Google Colab to PyCharm. |
| [Using-The-Trained-Weights](https://github.com/Rafif-AA/MMA3001-Final-Project/tree/Using-The-Trained-Weights) | Code that uses previously trained model weights without repeating model training. |

### PyCharm migration

Migration to PyCharm was initially planned because the free Google Colab runtime limits repeatedly interrupted development. Paid GPU access was later purchased through Colab, allowing the project to continue there. The migration was therefore not pursued further, and Google Colab remained the main development and training environment. The `PyCharm-Migration-1` branch preserves that earlier migration work.

### Using trained weights

The `Using-The-Trained-Weights` branch is intended for running code with an existing trained model rather than training a model from scratch. Supply the appropriate trained checkpoint, such as `best.pt`, and update the weight-file paths in the code before running it. Any input images or dataset files required by the code must also be available.

Use `main` to follow the training experiments described below, or switch to `Using-The-Trained-Weights` when reusing a model that has already been trained.

## Repository contents

The following files are on the `main` branch.

| File | Purpose |
| --- | --- |
| [MMA3001_Final_Project_Trial_1.ipynb](MMA3001_Final_Project_Trial_1.ipynb) | Baseline: Nano, 100 epochs, 640 x 640 |
| [MMA3001_Final_Project_Trial_2.ipynb](MMA3001_Final_Project_Trial_2.ipynb) | Nano, 100 epochs, 1280 x 1280 |
| [MMA3001_Final_Project_Trial_3.ipynb](MMA3001_Final_Project_Trial_3.ipynb) | Nano, up to 200 epochs, 640 x 640 |
| [MMA3001_Final_Project_Trial_4.ipynb](MMA3001_Final_Project_Trial_4.ipynb) | Small, 100 epochs, 640 x 640 |
| [MMA3001_Final_Project_Trial_5.ipynb](MMA3001_Final_Project_Trial_5.ipynb) | Small, up to 200 epochs, 1280 x 1280 |
| [LICENSE](LICENSE) | GNU Affero General Public License version 3 |

The notebooks contain model training, result visualisation, and activation analysis using PyTorch forward hooks. Trials 4 and 5 also contain cells for downloading run outputs and copying a checkpoint to Google Drive.

The dataset, trained checkpoints, and final report are supplied separately; they are not included in this repository. HTML source documentation and an automated software test suite/report have not yet been added.

## Dataset

The project considers four defect classes:

- Loose meat
- Packaging error
- Twisted meat
- Unsealed packaging

The dataset split reported in the final report is:

| Split | Images | Approximate proportion |
| --- | ---: | ---: |
| Training | 1,746 | 90% |
| Validation | 115 | 6% |
| Test | 76 | 4% |
| **Total** | **1,937** | **100%** |

Images and YOLO-format bounding-box annotations must be available before training. The notebooks currently expect the dataset configuration at:

```text
/content/drive/MyDrive/DATASET 3 - Pork Rasher/data.yaml
```

A typical folder layout is:

```text
DATASET 3 - Pork Rasher/
|-- data.yaml
|-- train/
|   |-- images/
|   `-- labels/
|-- valid/
|   |-- images/
|   `-- labels/
`-- test/
    |-- images/
    `-- labels/
```

Use the original dataset's `data.yaml` and class-ID mapping. Class names must appear in the same order as the numeric IDs in the annotation files. Update its paths to match your Drive folders. Dataset access and redistribution remain subject to the dataset owner's terms.

## Experimental design

The detector learns from labelled images through iterative optimisation of its weights using backpropagation. The experiments vary three settings: model size, maximum training epochs, and input resolution.

| Trial | Model | Initial weights | Maximum epochs | Input size | Epochs run, reported |
| --- | --- | --- | ---: | --- | ---: |
| 1 | YOLO26 Nano | `yolo26n.pt` | 100 | 640 x 640 | 100 |
| 2 | YOLO26 Nano | `yolo26n.pt` | 100 | 1280 x 1280 | 100 |
| 3 | YOLO26 Nano | `yolo26n.pt` | 200 | 640 x 640 | 139 |
| 4 | YOLO26 Small | `yolo26s.pt` | 100 | 640 x 640 | 100 |
| 5 | YOLO26 Small | `yolo26s.pt` | 200 | 1280 x 1280 | 179 |

The final report specifies early stopping with a patience of 100 epochs. Trials 3 and 5 stopped before their maximum epoch limits.

Evaluation considers precision-recall curves, mAP50, recall-confidence curves, precision-confidence curves, normalised confusion matrices, inference time, and training/validation loss curves.

## Reported results

The following values summarise the separately supplied **Final Report MMA3001.pdf**, particularly Sections 3 and 5. They have not been regenerated from the repository in a fresh run.

| Trial | Overall mAP50 | Approximate mean class recall from confusion matrix | Inference time (ms/image) | Weighted selection score |
| --- | ---: | ---: | ---: | ---: |
| 1 | 62.3% | 49.75% | 17.98 | 2.15 |
| **2** | **65.5%** | **82.00%** | **17.64** | **3.95** |
| 3 | 65.6% | 68.75% | 21.17 | 3.65 |
| 4 | 64.5% | 83.50% | 42.21 | 3.65 |
| 5 | 56.3% | 65.50% | 43.15 | 1.60 |

Here, mAP50 measures average precision across classes at an intersection-over-union threshold of 0.50. The mean class recall values are arithmetic means of the approximate normalised confusion-matrix diagonals. They should not be treated as recall at an arbitrary confidence threshold or as interchangeable with mAP50.

The report's decision matrix uses ratings from 1 to 5 and these weights:

| Criterion | Weight |
| --- | ---: |
| Recall across defect classes | 50% |
| Overall detection performance, mAP50 | 25% |
| Detection precision | 15% |
| Inference time | 10% |

Trial 2 ranks first under these weights. Trial 3 has the highest reported mAP50, but its advantage over Trial 2 is only 0.1 percentage point. Trial 4 has the highest approximate mean class recall. Repeated experiments would be needed to establish whether small differences are reliable.

Unsealed packaging is generally the strongest class, while packaging-error detection remains a priority for improvement. The report also identifies possible overfitting as training losses decrease while validation performance plateaus or declines.

**Timing note:** the inference times above follow Table 3.2. The paragraph below that table reverses the Trial 4 and Trial 5 values, and the decision matrix's timing ratings also favour Trial 5 over Trial 4. The selection scores above are reproduced as reported rather than recalculated.

## Running the notebooks in Google Colab

### 1. Open a trial

Open the relevant notebook from the repository and select its **Open in Colab** badge. Trial 2 is the configuration selected in the report.

Choose a GPU runtime in Colab. Available GPU hardware and memory vary; larger models and 1280 x 1280 inputs require more resources.

### 2. Install dependencies

Run the installation cell in the notebook. If the supporting packages are unavailable, install them in a Colab cell:

```python
%pip install ultralytics matplotlib pandas pillow
```

The notebooks use Python, Ultralytics, PyTorch, Matplotlib, pandas, and Pillow. Google Colab provides the `google.colab` integration.

Dependencies are currently unpinned. Saved notebook logs show different Ultralytics versions across trials, so this installation command is a starting point rather than an exact recreation of the final report's environment.

### 3. Mount Drive before training

Run this cell **before** any cell that accesses the dataset:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Place the dataset in Drive, verify the paths in `data.yaml`, and update the training cell if your dataset folder differs from the original location.

### 4. Train the model

Run the model-loading and training cells in order. For the selected Trial 2 configuration, the core training call is:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.train(
    data="/content/drive/MyDrive/DATASET 3 - Pork Rasher/data.yaml",
    epochs=100,
    imgsz=1280,
    patience=100,
)
```

Training time depends on the GPU, dataset access speed, model size, and image resolution. Initial model weights may be downloaded by Ultralytics.

### 5. Locate outputs and run analysis

Read the actual output directory printed by training. Update `results_dir`, the checkpoint passed to `YOLO(...)`, and any checkpoint-copy paths to refer to **that same run** before executing the plotting and activation-analysis cells.

Expected training outputs include:

- `weights/best.pt` and `weights/last.pt`
- `results.png` and training metrics
- Precision, recall, F1, and precision-recall curves
- Confusion matrices

Output names can vary with the installed Ultralytics version. Save required outputs to Drive or download them before the Colab runtime ends.

For activation analysis, choose the dummy-input resolution deliberately. The current analysis cells use 640 x 640 inputs, including in the notebooks trained at 1280 x 1280.

## Limitations and future work

The models have been evaluated using the project dataset but have not yet been tested under real-world production-line conditions. Their performance in an operating food-packaging facility therefore remains unverified.

Practical deployment may introduce changes in lighting, camera position, conveyor speed, motion blur, packaging materials, and product appearance. These conditions could affect detection accuracy, missed-defect rates, false alarms, and processing time compared with the reported experimental results.

Future work should evaluate the selected model on new images and video captured from an actual production line, including a wider range of defects and operating conditions. Confidence thresholds should be tuned to balance missed defects against false alarms, and inference speed should be measured on the intended deployment hardware. Further testing and refinement are needed before the model can be considered ready for routine industrial inspection.

## Acknowledgements and AI usage

This project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and its pretrained model weights. Training and analysis were carried out using Google Colab and PyTorch.

As disclosed in the final report, ChatGPT generated most of the project code, with author modifications mainly to file paths, chart titles, and other minor settings. AI assistance was also used to draft and refine report text, present evaluation results, and organise the report under the author's supervision. This README was prepared with AI assistance from that report and a review of the committed notebooks.

## License

Copyright (c) 2026 Rafif-AA.

The project code is licensed under the **GNU Affero General Public License version 3.0 (AGPL-3.0)**. See [LICENSE](LICENSE) for the full terms.

Ultralytics YOLO is used under its AGPL-3.0 open-source terms; see the [Ultralytics licensing information](https://www.ultralytics.com/license). Third-party libraries, pretrained weights, and dataset materials retain their applicable licences and ownership. This repository's licence does not grant rights to separately supplied dataset materials.
