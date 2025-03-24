from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader

from config import CHECKPOINT_PATH, TEST_SET_LIST, TRAIN_LABEL_LIST, ROOT
from dataloader import CustomDataset
from model_256 import Model


def forward_pass(model, x, device):
    model.eval()
    for i in range(len(x)):
        x[i] = x[i].to(device)
    return model(x)


def test():
    weight_file_path = CHECKPOINT_PATH / "CaH.pt"
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device_name}")
    device = torch.device(device_name)
    model = Model().to(device)
    model.load_state_dict(torch.load(weight_file_path, map_location=device))
    print(f"Model weights loaded in {device_name}")

    dataloader = DataLoader(
        dataset=CustomDataset(
            pid_path=TEST_SET_LIST,
            label_path=TRAIN_LABEL_LIST,
        ),
        batch_size=1
    )

    print("Test started")
    with open("/home/julia/Research/AffinityPred/Results/YPred_core2016.lst", "w") as file:
        with torch.no_grad():
            for x, y in dataloader:
                prediction = forward_pass(model, x, device)
                prediction = prediction.cpu().detach().numpy()
                for value in prediction[0]:
                    file.write(f"{round(value, 3)}\n")
    print("Test done")


if __name__ == "__main__":
    test()
