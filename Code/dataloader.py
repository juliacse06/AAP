import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from config import LL_LENGTH, P_LENGTH, A_LENGTH, P_FDIM, LL_FEATURE_PATH, ANGLE_FEATURE_PATH, PP_FEATURE_PATH


class CustomDataset(Dataset):
    def __init__(self, pid_path: Path, label_path: Path):
        #print("Loading data")
        all_pids: list = np.loadtxt(fname=str(pid_path.absolute()), dtype='str').tolist()
        y_all: list = np.loadtxt(fname=str(label_path.absolute()), dtype='float').tolist()
        #print(len(all_pids), len(y_all))

        self.ll_data = np.zeros((len(all_pids), LL_LENGTH))  # ll_info['smile_features'].shape[0]
        self.pp_data = np.zeros((len(all_pids), P_LENGTH, P_FDIM))
        self.angle_data = np.zeros((len(all_pids), A_LENGTH))
        self.y_labels = []

        for i, pid in enumerate(all_pids):
            with open(f"{ANGLE_FEATURE_PATH.absolute()}/{pid}_angle_info.pkl", "rb") as dif:
                angle_info = pickle.load(dif)   #pl_angle Dim 40 is ok or pl_bin_angle
            #das_2p2l_bin_angle, dash_2p2l_bin_angle, da_3p1l_bin_angle,  hydro_2p2l_bin_angle,  hydros_2p2l_bin_angle  
            # cac_2p2l_bin_angle, cah_2p2l_bin_angle
            if angle_info['cah_3p1l_bin_angle'].shape[0] > A_LENGTH: #dash_2p2l_bin_angle   dash_3p1l_bin_angle
                self.angle_data[i, :] = angle_info['cah_3p1l_bin_angle'][:A_LENGTH]
            else:
                self.angle_data[i, :angle_info['cah_3p1l_bin_angle'].shape[0]] = angle_info['cah_3p1l_bin_angle']

            
            with open(f"{LL_FEATURE_PATH.absolute()}/{pid}_ll_info.pkl", "rb") as dif:
                ll_info = pickle.load(dif)
            self.ll_data[i] = ll_info['smile_features'][:LL_LENGTH]

            with open(f"{PP_FEATURE_PATH.absolute()}/{pid}_pp_info.pkl", "rb") as dif:
                pp_info = pickle.load(dif)
            if pp_info['aa7pcpHHm_feature'].shape[0] > P_LENGTH:
                self.pp_data[i, :, :] = pp_info['aa7pcpHHm_feature'][:P_LENGTH, :]
            else:
                self.pp_data[i, :pp_info['aa7pcpHHm_feature'].shape[0], :] = pp_info['aa7pcpHHm_feature'][:,:]
            #print(i)

            self.y_labels.append(y_all[i])
            #print(i)

    def __getitem__(self, idx):
        pp = np.float32(self.pp_data[idx, :])
        al = np.float32(self.angle_data[idx, :])
        ll = np.int32(self.ll_data[idx, :])
        return (pp, al, ll), self.y_labels[idx]

    def __len__(self):
        return len(self.y_labels)
