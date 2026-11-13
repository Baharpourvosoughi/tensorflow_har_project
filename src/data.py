from pathlib import Path
import zipfile
import urllib.request

import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/raw")
ZIP_PATH = DATA_DIR / "uci_har.zip"
EXTRACT_DIR = DATA_DIR / "UCI HAR Dataset"

# Official UCI archive. The archive is about 76 MB; it is downloaded once.
DATA_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def ensure_dataset():
    """Download and extract the public UCI HAR dataset if it is not present."""
    train_file = EXTRACT_DIR / "train" / "X_train.txt"
    if train_file.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print("Downloading UCI HAR dataset (~76 MB). This may take a few minutes...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(DATA_DIR)
    if not train_file.exists():
        raise FileNotFoundError(
            "Dataset extraction did not produce the expected UCI HAR Dataset/train/X_train.txt"
        )


def load_split(split):
    folder = EXTRACT_DIR / split
    X = np.loadtxt(folder / f"X_{split}.txt", dtype=np.float32)
    y = np.loadtxt(folder / f"y_{split}.txt", dtype=np.int64) - 1
    return X, y


def load_data():
    ensure_dataset()
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    # UCI HAR already provides a fixed train/test split.
    # Fit scaling only on training data to avoid test-set leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, y_train, X_test, y_test, scaler
