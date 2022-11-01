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
from sklearn.metrics import accuracy_score
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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)


HIDDEN_SIZE = 360
NUM_ATTENTION_HEADS = 3
BATCH_SIZE = 32

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
        video_features_path = os.path.join(self.folder, "features_video_GRU", self.vid_files[vid_ind].replace(".mp4", ".npy"))
        #video_features_path = os.path.join("features_pretrained", self.vid_files[vid_ind].replace(".mp4", ".npy"))
        wav_features_path = os.path.join(self.folder, "features_wav2vec_all", self.vid_files[vid_ind].replace(".mp4", ".npy"))
        video_features = np.load(video_features_path)
        wav_features = np.load(wav_features_path)
        video_features = torch.tensor(video_features)
        wav_features = torch.tensor(wav_features)
        class_id = self.target[self.vid_files[vid_ind].replace("mp4", "flv")]

        return video_features, wav_features, class_id, self.vid_files[vid_ind]

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, hidden_size=HIDDEN_SIZE, num_atten=NUM_ATTENTION_HEADS):
        super().__init__()
        self.inter_dim = hidden_size//num_atten
        self.num_heads = num_atten
        self.fc_q = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_k = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_v = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)   
        self.multihead_attn = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                    self.num_heads,
                                                    dropout = 0.0,
                                                    bias = True)                   
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps = 1e-6)
        self.fc = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
    
    def forward(self, mod):
        q = self.fc_q(mod)
        k = self.fc_k(mod)
        v = self.fc_v(mod)
        
        self_atten = self.multihead_attn(q, k, v, need_weights = False)[0]
        mod_q = self.dropout(self.fc(self_atten))
        
        mod_q += mod
        
        mod_q = self.layer_norm(mod_q)

        return mod_q

class CrossAttentionModel(nn.Module):
    def __init__(self, hidden_dim, hidden_size=HIDDEN_SIZE, num_atten=NUM_ATTENTION_HEADS):
        super().__init__()
        self.inter_dim = hidden_size//num_atten
        self.num_heads = num_atten
        self.fc_audq = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_audk = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_audv = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)  
        
        self.fc_vidq = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_vidk = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_vidv = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)

        # self.fc_audq_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        # self.fc_audk_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        # self.fc_audv_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_vidq_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_vidk_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)
        self.fc_vidv_s = nn.Linear(hidden_dim, self.inter_dim*self.num_heads)

        self.multihead_attn_vid = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.5,
                                                         bias = True)
        # self.multihead_attn_aud = nn.MultiheadAttention(self.inter_dim*self.num_heads,
        #                                                  self.num_heads,
        #                                                  dropout = 0.5,
        #                                                  bias = True)
        
        # self.multihead_attn_selfaud = nn.MultiheadAttention(self.inter_dim*self.num_heads,
        #                                                  self.num_heads,
        #                                                  dropout = 0.5,
        #                                                  bias = True)
        self.multihead_attn_selfvid = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                         self.num_heads,
                                                         dropout = 0.5,
                                                         bias = True)                      
        self.dropout = nn.Dropout(0.5)

        self.layer_norm_v = nn.LayerNorm(hidden_dim, eps = 1e-6)
        # self.layer_norm_a = nn.LayerNorm(hidden_dim, eps = 1e-6)
        self.layer_norm_v_s = nn.LayerNorm(hidden_dim, eps = 1e-6)
        # self.layer_norm_a_s = nn.LayerNorm(hidden_dim, eps = 1e-6)

        self.fc_video = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
        # self.fc_audio = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_video_s = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
        # self.fc_audio_s = nn.Linear(self.inter_dim*self.num_heads, hidden_dim)
    
    def forward(self, audio, video):
        video_q = self.fc_vidq(video)
        video_k = self.fc_vidk(video)
        video_v = self.fc_vidv(video)
        audio_q = self.fc_audq(audio)
        audio_k = self.fc_audk(audio)
        audio_v = self.fc_audv(audio)
        video_cross = self.multihead_attn_vid(video_q, audio, video_v, need_weights = False)[0]
        # audio_cross = self.multihead_attn_aud(audio_q, video_k, video_v, need_weights = False)[0]
        video_q = self.dropout(self.fc_video(video_cross))
        # audio_q = self.dropout(self.fc_audio(audio_cross))
        video_q += video
        # audio_q += audio
        video_q = self.layer_norm_v(video_q)
        # audio_q = self.layer_norm_a(audio_q)
        vid_q_s = self.fc_vidq_s(video_q)
        vid_k_s = self.fc_vidk_s(video_q)
        vid_v_s = self.fc_vidv_s(video_q)
        # aud_q_s = self.fc_audq_s(audio_q)
        # aud_k_s = self.fc_audk_s(audio_q)
        # aud_v_s = self.fc_audv_s(audio_q)
        vid_self = self.multihead_attn_selfvid(vid_q_s, vid_k_s, vid_v_s, need_weights = False)[0]
        # aud_self = self.multihead_attn_selfaud(aud_q_s, aud_k_s, aud_v_s, need_weights = False)[0]
        vid_q_s = self.dropout(self.fc_video_s(vid_self))
        # aud_q_s = self.dropout(self.fc_audio_s(aud_self))
        vid_q_s += video_q
        # aud_q_s += audio_q
        vid_q_fin = self.layer_norm_v_s(vid_q_s)
        # aud_q_fin = self.layer_norm_a_s(aud_q_s)
        #return video_q, audio_q
        return vid_q_fin
'''
class MultimodalClassifier(nn.Module):
    def __init__(self, seq_len_vid, seq_len_aud, hidden_dim, num_layers):
        super().__init__()
        self.hid = 120
        #self.fc_1 = nn.Linear(self.hid*3, self.hid)
        self.fc = nn.Linear(self.hid*2, 6)
        self.fc_aud = nn.Linear(hidden_dim, self.hid)
        self.fc_vid = nn.Linear(hidden_dim, self.hid)
        self.units = nn.ModuleList()
        self.self_aud_units = nn.ModuleList()
        self.self_vid_units = nn.ModuleList()
        for ind in range(num_layers):
            self.units.append(CrossAttentionModel(self.hid))
        self.vid_cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.aud_cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.layer_norm = nn.LayerNorm(self.hid*2, eps = 1e-6)
        
        # for ind in range(num_layers):
        #     self.self_aud_units.append(SelfAttention(hidden_dim))
        
        for ind in range(num_layers):
            self.self_vid_units.append(SelfAttention(hidden_dim))
        self.pos_encoder_cross_aud = PositionalEncoding(self.hid)
        self.pos_encoder_cross_vid = PositionalEncoding(self.hid)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        # self.pos_encoder_aud = PositionalEncoding(hidden_dim)
        # self.pos_encoder_vid = PositionalEncoding(hidden_dim)
        # self.fc_aud_self = nn.Linear(hidden_dim, self.hid)
        self.fc_vid_self = nn.Linear(hidden_dim, self.hid)
        
    def forward(self, video_orig, audio_orig):
        cls_tokens_vid = self.vid_cls_token.expand(video_orig.size(0), -1, -1)
        cls_tokens_aud = self.aud_cls_token.expand(audio_orig.size(0), -1, -1)
        #audio_orig = torch.cat((cls_tokens_aud, audio_orig), 1)
        #video_orig = torch.cat((cls_tokens_vid, video_orig), 1)
        audio = audio_orig.permute(1, 0, 2)
        video = video_orig.permute(1, 0, 2)
        video = self.fc_vid(video)
        audio = self.fc_aud(audio)
        # video += self.pos_encoder_cross_vid(video)
        #audio += self.pos_encoder_cross_aud(audio)
        # aud_self = audio_orig.permute(1, 0, 2)
        vid_self = video_orig.permute(1, 0, 2)
        # aud_self += self.pos_encoder_aud(aud_self)
        # vid_self += self.pos_encoder_vid(vid_self)

        # for model_ca in self.units:
        #     video, audio = model_ca(audio, video)
        # for model_ca in self.units:
        #     video = model_ca(audio, video)

        # for model_aud_self in self.self_aud_units:
        #     aud_self = model_aud_self(aud_self)
        
        for model_vid_self in self.self_vid_units:
            vid_self = model_vid_self(vid_self)

        audio = audio.permute(1, 0, 2)
        video = video.permute(1, 0, 2)
        # video_new = self.fc_vid_self(video_orig)
        # video_new = torch.mean(video_new, 1).squeeze(1)
        # aud_self = aud_self.permute(1, 0, 2)
        vid_self = vid_self.permute(1, 0, 2)
        # aud_self = self.fc_aud_self(aud_self)
        vid_self = self.fc_vid_self(vid_self)
        # audio_repr = audio[:, 0, :]
        # video_repr = video[:, 0, :]
        audio = torch.mean(audio, 1).squeeze(1)
        video = torch.mean(video, 1).squeeze(1)
        # aud_self = torch.mean(aud_self, 1).squeeze(1)
        vid_self = torch.mean(vid_self, 1).squeeze(1)
        #concat = audio + video + vid_self
        #concat = torch.cat((audio, video, aud_self, vid_self), -1)
        concat = torch.cat((audio, video), -1)
        concat = self.layer_norm(concat)
        #concat = audio_repr*video_repr
        out = self.fc(concat)
        #out = self.fc(out)
        return out'''

class MultimodalClassifier(nn.Module):
    def __init__(self, seq_len_vid, seq_len_aud, hidden_dim, num_layers):
        super().__init__()
        self.hid = 360
        #self.fc_1 = nn.Linear(self.hid*3, self.hid)
        self.fc = nn.Linear(self.hid*3, 6)
        self.fc_aud = nn.Linear(hidden_dim, self.hid)
        self.fc_vid = nn.Linear(hidden_dim, self.hid)
        self.units = nn.ModuleList()
        self.self_aud_units = nn.ModuleList()
        self.self_vid_units = nn.ModuleList()
        for ind in range(num_layers):
            self.units.append(CrossAttentionModel(self.hid))
        self.vid_cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.aud_cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.layer_norm = nn.LayerNorm(self.hid*3, eps = 1e-6)
        
        # for ind in range(num_layers):
        #     self.self_aud_units.append(SelfAttention(hidden_dim))
        
        for ind in range(num_layers):
            self.self_vid_units.append(SelfAttention(hidden_dim))
        self.pos_encoder_cross_aud = PositionalEncoding(self.hid)
        self.pos_encoder_cross_vid = PositionalEncoding(self.hid)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        # self.pos_encoder_aud = PositionalEncoding(hidden_dim)
        # self.pos_encoder_vid = PositionalEncoding(hidden_dim)
        # self.fc_aud_self = nn.Linear(hidden_dim, self.hid)
        self.fc_vid_self = nn.Linear(hidden_dim, self.hid)
        
    def forward(self, video_orig, audio_orig):
        cls_tokens_vid = self.vid_cls_token.expand(video_orig.size(0), -1, -1)
        cls_tokens_aud = self.aud_cls_token.expand(audio_orig.size(0), -1, -1)
        #audio_orig = torch.cat((cls_tokens_aud, audio_orig), 1)
        #video_orig = torch.cat((cls_tokens_vid, video_orig), 1)
        audio = audio_orig.permute(1, 0, 2)
        video = video_orig.permute(1, 0, 2)
        video = self.fc_vid(video)
        audio = self.fc_aud(audio)
        # video += self.pos_encoder_cross_vid(video)
        #audio += self.pos_encoder_cross_aud(audio)
        # aud_self = audio_orig.permute(1, 0, 2)
        vid_self = video_orig.permute(1, 0, 2)
        # aud_self += self.pos_encoder_aud(aud_self)
        # vid_self += self.pos_encoder_vid(vid_self)

        # for model_ca in self.units:
        #     video, audio = model_ca(audio, video)
        for model_ca in self.units:
            video = model_ca(audio, video)

        # for model_aud_self in self.self_aud_units:
        #     aud_self = model_aud_self(aud_self)
        
        for model_vid_self in self.self_vid_units:
            vid_self = model_vid_self(vid_self)

        audio = audio.permute(1, 0, 2)
        video = video.permute(1, 0, 2)
        # video_new = self.fc_vid_self(video_orig)
        # video_new = torch.mean(video_new, 1).squeeze(1)
        # aud_self = aud_self.permute(1, 0, 2)
        vid_self = vid_self.permute(1, 0, 2)
        # aud_self = self.fc_aud_self(aud_self)
        vid_self = self.fc_vid_self(vid_self)
        # audio_repr = audio[:, 0, :]
        # video_repr = video[:, 0, :]
        audio = torch.mean(audio, 1).squeeze(1)
        video = torch.mean(video, 1).squeeze(1)
        # aud_self = torch.mean(aud_self, 1).squeeze(1)
        vid_self = torch.mean(vid_self, 1).squeeze(1)
        #concat = audio + video + vid_self
        #concat = torch.cat((audio, video, aud_self, vid_self), -1)
        concat = torch.cat((audio, video, vid_self), -1)
        concat = self.layer_norm(concat)
        #concat = audio_repr*video_repr
        out = self.fc(concat)
        #out = self.fc(out)
        return out


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
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=False)
    return loader

def compute_loss(output, labels):
    loss = nn.CrossEntropyLoss(reduction='mean')(output, labels.long())
    # pt = torch.exp(-ce_loss)
    # loss = ((1-pt)**0 * ce_loss).mean()
    return loss

def train(ind):
    train_loader = create_dataset("train", ind)
    val_loader = create_dataset("val", ind)
    model = MultimodalClassifier(1568, 249, 768, 2)
    model.to(device)
    base_lr = 1e-4
    optimizer = Adam([{'params':model.parameters(), 'lr':base_lr}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    final_val_loss = 99999
    for e in range(20):
        print('Epoch-{0} lr: {1}'.format(e, optimizer.param_groups[0]['lr']))
        model.train()
        tot_loss = 0.0
        val_loss = 0.0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        for i, data in enumerate(tqdm(train_loader)):
            model.zero_grad()
            video_features, wav_features, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            # vid_cls = torch.zeros(len(labels), 1, 768).to(device)
            # aud_cls = torch.zeros(len(labels), 1, 768).to(device)
            # video_features = torch.cat((vid_cls, video_features), 1)
            # wav_features = torch.cat((aud_cls, wav_features), 1)
            pred = model(video_features, wav_features.float())
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
        #scheduler.step()
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                video_features, wav_features, labels = data[0].to(device), data[1].to(device), data[2].to(device)
                # vid_cls = torch.zeros(len(labels), 1, 768).to(device)
                # aud_cls = torch.zeros(len(labels), 1, 768).to(device)
                # video_features = torch.cat((vid_cls, video_features), 1)
                # wav_features = torch.cat((aud_cls, wav_features), 1)
                pred = model(video_features, wav_features.float())
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
                        'seq_multimodal_model' + str(ind) + '.tar')
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
