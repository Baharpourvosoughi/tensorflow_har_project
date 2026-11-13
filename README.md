# Human Activity Recognition with TensorFlow/Keras

A compact, reproducible multi-class classification project using the public **UCI Human Activity Recognition Using Smartphones** dataset.

The model classifies six activities from smartphone inertial-sensor features:

- Walking
- Walking upstairs
- Walking downstairs
- Sitting
- Standing
- Laying

Dataset: UCI Machine Learning Repository, dataset 240. The dataset was collected from 30 volunteers using a waist-mounted smartphone. It contains 10,299 pre-segmented examples and 561 engineered features per example. The official archive is approximately 76 MB.

Dataset page: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones  
DOI: https://doi.org/10.24432/C54G7M  
License: CC BY 4.0. Please retain attribution when sharing derived work.

## Requirements

- Python 3.10–3.12
- TensorFlow 2.16–2.20
- A CPU is sufficient; no paid API or GPU is required.

On Apple Silicon Macs, TensorFlow installation depends on the macOS/Python wheel available for your environment. If `pip install -r requirements.txt` reports that no TensorFlow distribution is available, send the full error before trying random package substitutions.

## Setup (macOS / Linux)

Extract the project ZIP, open Terminal in the project folder, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and evaluate

```bash
python -m src.train
```

On the first run, the script downloads the official UCI archive (~76 MB) into `data/raw/`, then extracts it. Later runs reuse the downloaded data.

The script:
1. Loads the official train/test split.
2. Fits `StandardScaler` on training data only.
3. Creates a validation split from the training set.
4. Builds and trains a Keras Sequential neural network.
5. Uses EarlyStopping and ModelCheckpoint.
6. Evaluates once on the held-out test set.
7. Saves the model, metrics, classification report, and confusion matrix.

## Run a sample prediction

```bash
python -m src.predict
```

This demonstrates prediction on one sample from the official test set. It is not a live sensor application.

## Run tests

```bash
python -m pytest
```

## Project structure

```text
tensorflow_har_project/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── evaluate.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_model.py
├── data/
│   └── raw/                 # downloaded automatically; not committed
├── models/                  # trained model and scaler parameters
├── reports/
│   └── figures/
├── requirements.txt
├── pytest.ini
└── README.md
```

## Important methodological notes

- The dataset already provides a fixed training and test split; this project preserves it.
- A validation subset is created only from the training split.
- The scaler is fitted only on training data, preventing test-set leakage.
- The six labels are encoded as integers 0–5 for Keras sparse categorical cross-entropy.
- This is a learning project, not a real-time activity detector or a health/medical assessment system.
- The dataset's precomputed 561-feature representation is used here to focus on learning TensorFlow/Keras. It does not train directly on raw sensor sequences.
