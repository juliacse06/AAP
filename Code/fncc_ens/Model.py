import torch
from torch import nn
import torch.nn.functional as F

class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
       
    def forward(self, data):
        cah, hydroS, dasp = data
        inputs = [x.to(torch.float32).unsqueeze(1) for x in data]
        concat = torch.cat(inputs, dim=1)
        output = self.classifier(concat)
        return output
    