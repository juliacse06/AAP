from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from FC_config import FEATURE_PATH


class CustomDataset(Dataset):
    def __init__(self, pid_path: Path, label_path: Path):
        #print("Loading data")
        all_pids: list = np.loadtxt(fname=str(pid_path.absolute()), dtype='str').tolist()
        y_all: list = np.loadtxt(fname=str(label_path.absolute()), dtype='float').tolist()
        
        cah_all: list = np.loadtxt(FEATURE_PATH / "CAH_train.lst", dtype='float').tolist()
        hydroS_all: list = np.loadtxt(FEATURE_PATH / "HydroS_train.lst", dtype='float').tolist()
        dash_all: list = np.loadtxt(FEATURE_PATH / "DASH_train.lst", dtype='float').tolist()
                
        
        numberOfFeatures = 3

        self.feature = np.zeros((len(all_pids), numberOfFeatures))  # ll_info['smile_features'].shape[0]
        self.y_labels = []
        
        # predicted affiity scores
        self.CaH_angles=[]
        self.HydroS_angles=[]
        self.DASH_angles=[]


        for i, pid in enumerate(all_pids):
            self.y_labels.append(y_all[i])
            
            self.CaH_angles.append(cah_all[i])
            self.HydroS_angles.append(hydroS_all[i]) 
            self.DASH_angles.append(dash_all[i])           


    def __getitem__(self, idx):
        cah = np.float32(self.CaH_angles[idx])
        hydroS = np.float32(self.HydroS_angles[idx])
        dasp = np.float32(self.DASH_angles[idx])
   
    
        return (cah, hydroS, dasp), self.y_labels[idx]

    def __len__(self):
        return len(self.y_labels)
