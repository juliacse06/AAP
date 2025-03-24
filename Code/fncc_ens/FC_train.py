import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from FC_config import (
    FEATURE_PATH,
    TRAIN_SET_LIST,
    TRAIN_LABEL_LIST,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    TOTAL_EPOCH,
)
from FC_dataloader import CustomDataset
from Model import EnsembleModel


def train(i):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device_name}")
    device = torch.device(device_name)
    model = EnsembleModel().to(device)
    print(f"Model instance created and loaded in {device_name}")

    print("Starting training process")

    dataloader = DataLoader(
        dataset=CustomDataset(
            pid_path=TRAIN_SET_LIST,
            label_path=TRAIN_LABEL_LIST,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    optimizer = optim.AdamW(model.parameters())
    #loss_function = LogCoshLoss()
    loss_function = nn.L1Loss()

    best_epoch = -1
    best_loss = 100000000
    for n_epoch in range(1, TOTAL_EPOCH + 1):
        print(f"Starting epoch {n_epoch} / {TOTAL_EPOCH}")
        best_val_loss = 100000000
        for X, Y in dataloader:
            loss = forward_pass(model, X, Y, device, optimizer, loss_function)
            if loss < best_val_loss:
                best_val_loss = loss
        
        if best_val_loss < best_loss:
            best_loss = best_val_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH / f"FCens_{i}_{n_epoch}.pt")           

    print(f"Training finish. Best epoch {best_epoch}")


def forward_pass(model, x, y, device, optimizer, loss_function):
    model.train()

    for i in range(len(x)):
        x[i] = x[i].to(device)
    y = y.to(device)

    optimizer.zero_grad()
    output = model(x)
    loss = loss_function(output.view(-1), y.view(-1))

    loss.backward()
    optimizer.step()

    return loss.item() / len(y)


if __name__ == "__main__":
    for i in range(0,100):
        train(i)
