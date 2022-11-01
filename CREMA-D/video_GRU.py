import sys
import os
import logging
import torch
import numpy as np
import json
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.metrics import f1_score
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
SEED = 9348
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class MyDataset(Dataset):
    def __init__(self, folder, target_dict):
        self.folder = folder
        self.target = target_dict
        vid_files = list(target_dict.keys())
        vid_files = [x.replace("flv", "mp4") for x in vid_files]
        vid_files = [x for x in vid_files if ".mp4" in x]
        vid_files = [x for x in vid_files if '1076_MTI_SAD_XX.mp4' not in x]
        self.vid_files = vid_files

    def __len__(self):
        return len(self.vid_files) 
        
    def __getitem__(self, vid_ind):
        video_features_path = os.path.join(self.folder, "features_timesformer_all", self.vid_files[vid_ind].replace(".mp4", ".npy"))
        video_features = np.load(video_features_path)
        video_features = torch.tensor(video_features)
        class_id = self.target[self.vid_files[vid_ind].replace("mp4", "flv")]

        return video_features, class_id, self.vid_files[vid_ind]

class SelfAttentionModel(nn.Module):
    def __init__(self, hidden_dim, hidden_size, num_atten):
        super().__init__()
        self.inter_dim = hidden_size//num_atten
        self.num_heads = num_atten
        self.fc_q = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_k = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_v = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)        
        self.multihead_attn = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                    self.num_heads,
                                                    dropout = 0.1,
                                                    bias = True)
        
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps = 1e-6)
        self.fc = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
        
    def forward(self, mod):
        q = self.fc_q(mod)
        k = self.fc_k(mod)
        v = self.fc_v(mod)
        self_atten = self.multihead_attn(q, k, v,  need_weights = False)[0]
        mod_q = self.fc(self_atten)
        mod_q += mod
        mod_q = self.layer_norm(mod_q)        
        return mod_q

class GRUModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, layers, bidirectional_flag):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.num_layers = layers
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=layers, bidirectional=bidirectional_flag, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.units = nn.ModuleList()
        for ind in range(2):
            self.units.append(SelfAttentionModel(hidden_dim*2, 360, 3))

    def forward(self, x):

        hidden_seq, hidden = self.rnn(x)
        permuted_x = hidden_seq.permute(1, 0, 2)
        for attn_model in self.units:
            permuted_x = attn_model(permuted_x)
        output_con = permuted_x.permute(1, 0, 2)
        hidden = torch.mean(output_con, 1).squeeze(1)
        # if self.rnn.bidirectional:
        #     hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        # else:
        #     hidden = self.dropout(hidden[-1,:,:])
        out = self.fc(hidden)
        return output_con, out

def create_dataset(mode, ind):
    folder = "split_" + str(ind)
    if mode == 'train':
        f = open("splits/train_labels" + str(ind) + ".json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        f = open("splits/val_labels" + str(ind) + ".json")
        labels = json.load(f)
        f.close()
    else:
        f = open("splits/test_labels" + str(ind) + ".json")
        labels = json.load(f)
        f.close()
    dataset = MyDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=32,
                    shuffle=True,
                    drop_last=False)
    return loader

def compute_loss(output, labels):
    ce_loss = nn.CrossEntropyLoss(reduction='none')(output, labels.long())
    pt = torch.exp(-ce_loss)
    loss = ((1-pt)**0 * ce_loss).mean()
    return loss

def train(ind):
    train_loader = create_dataset("train", ind)
    val_loader = create_dataset("val", ind)
    model = GRUModel(768, 384, 6, 0.5, 2, True)
    model.to(device)
    base_lr = 5e-5
    optimizer = Adam([{'params':model.parameters(), 'lr':base_lr}])
    final_val_loss = 99999
    for e in range(20):
        model.train()
        tot_loss = 0.0
        val_loss = 0.0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        for i, data in enumerate(tqdm(train_loader)):
            model.zero_grad()
            video_features, labels, name = data[0].to(device), data[1].to(device), data[2]
            _, pred = model(video_features)
            loss = compute_loss(pred, labels)
            tot_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_class = torch.argmax(pred, dim = 1)
            pred_class = pred_class.detach().cpu().numpy()
            pred_class = list(pred_class)
            pred_tr.extend(pred_class)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                video_features, labels, name = data[0].to(device), data[1].to(device), data[2]
                _, pred = model(video_features)
                loss = compute_loss(pred, labels)
                val_loss += loss.item()
                pred_class = torch.argmax(pred, dim = 1)
                pred_class = pred_class.detach().cpu().numpy()
                pred_class = list(pred_class)
                pred_val.extend(pred_class)
                labels = labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val.extend(labels)
        if val_loss < final_val_loss:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),},
                        'context_video' + str(ind) + '.tar')
            final_val_loss = val_loss
        train_loss = tot_loss/len(train_loader)
        train_f1 = f1_score(gt_tr, pred_tr, average='weighted')
        val_loss_log = val_loss/len(val_loader)
        val_f1 = f1_score(gt_val, pred_val, average='weighted')
        e_log = e + 1
        logger.info(f"Epoch {e_log}, \
                    Training Loss {train_loss},\
                    Training Accuracy {train_f1}")
        logger.info(f"Epoch {e_log}, \
                    Validation Loss {val_loss_log},\
                    Validation Accuracy {val_f1}")

if __name__ == "__main__":
    train(2)
