from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader

from FC_config import CHECKPOINT_PATH, TEST_SET_LIST, TRAIN_LABEL_LIST, ROOT
from FC_dataloaderTest import CustomDataset
from Model import EnsembleModel

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


def forward_pass(model, x, device):
    model.eval()
    for i in range(len(x)):
        x[i] = x[i].to(device)
    return model(x)


def test(weight):
    weight_file_path = CHECKPOINT_PATH / "FCE.pt"
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = EnsembleModel().to(device)
    model.load_state_dict(torch.load(weight_file_path, map_location=device))
    #print(f"Model weights loaded in {device_name}")

    dataloader = DataLoader(
        dataset=CustomDataset(
            pid_path=TEST_SET_LIST,
            label_path=TRAIN_LABEL_LIST,
        ),
        batch_size=1
    )

    print("Test started")
    with open("/home/julia/Research/AffinityPred/Final_models/Angle/AAP/FCE_CSAR_36.lst", "w") as file:
        with torch.no_grad():
            for x, y in dataloader:
                prediction = forward_pass(model, x, device)
                prediction = prediction.cpu().detach().numpy()
                for value in prediction[0]:
                    file.write(f"{round(value, 3)}\n")
    print("Test done")


if __name__ == "__main__":
    test()

                
