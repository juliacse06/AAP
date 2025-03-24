from pathlib import Path


BATCH_SIZE = 128
TOTAL_EPOCH = 20
ROOT = Path("~/AffinityPred")
FEATURE_PATH = ROOT / "Final_models/Angle/AAP/TrainData"
TESTFEATURE_PATH = ROOT / "Final_models/Angle/AAP/Results"

TRAIN_SET_LIST = FEATURE_PATH / "train_20.lst"
TRAIN_LABEL_LIST = FEATURE_PATH / "Y_train_20.lst"

TEST_SET_LIST = ROOT / "CSARHIQ_36.lst"
TEST_LABEL_LIST = ROOT / ""

CHECKPOINT_PATH = ROOT / "Final_models/AAP/Weights"
