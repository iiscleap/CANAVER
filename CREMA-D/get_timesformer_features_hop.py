import os
import torch
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
from torchvision.io import read_video
import torch.nn as nn
import random
#from timesformer_pytorch import TimeSformer
from timesformer.models.vit import TimeSformer
from sklearn.metrics import f1_score
import gc
from tqdm import tqdm 

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

np.random.seed(1234)
torch.manual_seed(1234)

#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

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
        video_file = os.path.join(self.folder, self.vid_files[vid_ind])
        class_id = self.target[self.vid_files[vid_ind].replace("mp4", "flv")]
        frames, _, _ = read_video(video_file)
        tot_frames = 150
        frame_window, frame_hop = 8, 6
        frame_list = []
        if frames.shape[0] > tot_frames:
            start = np.random.randint(0, frames.shape[0]- tot_frames)
            final_sig = frames[start:start+tot_frames, :, :, :]
        else:
            pad_begin_len = random.randint(0, tot_frames - frames.shape[0])
            pad_end_len = tot_frames - frames.shape[0] - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((pad_begin_len, frames.shape[1], frames.shape[2], frames.shape[3]))
            pad_end = torch.zeros((pad_end_len, frames.shape[1], frames.shape[2], frames.shape[3]))
            final_sig = torch.cat((pad_begin, frames, pad_end), 0)
        seq_length = tot_frames//frame_hop
        for ind in range(seq_length):
            start_index = ind*frame_window
            end_index = start_index + frame_hop + 1
            end_index = min(tot_frames, end_index)
            frame_windowed = final_sig[start_index:end_index, :, :, :]
            if frame_windowed.shape[0] < frame_window:
                padding = torch.zeros((frame_window-frame_windowed.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
                frame_windowed = torch.cat((frame_windowed, padding), 0)
                frame_list.append(frame_windowed)
        video_sig = torch.cat([x.float() for x in frame_list])
        video_sig = video_sig.reshape((len(frame_list), frame_window, frames.shape[1], frames.shape[2], frames.shape[3]))
        return video_sig, class_id, self.vid_files[vid_ind]

def collate_fn(batch):
    batch_inp = []
    for ele in batch:
        batch_inp.append(ele)
    return batch_inp

def create_dataset(mode, ind):
    folder = "mp4"
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
                    batch_size=4,
                    shuffle=True,
                    drop_last=False)
    return loader

def get_features(ind):
    train_loader = create_dataset("train", ind)
    val_loader = create_dataset("val", ind)
    test_loader = create_dataset("test", ind)
    model = TimeSformer(img_size=224, num_classes=6,
                        num_frames=8, attention_type='divided_space_time')
    checkpoint = torch.load(os.path.join('split_' + str(ind), 'video_timesformer_pretrained_' + str(ind) + '.tar'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    if os.path.exists(os.path.join('split_' + str(ind), "features_timesformer_all")) == False:
        os.mkdir(os.path.join('split_' + str(ind), "features_timesformer_all"))
    with torch.no_grad():
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels, videos = data[0].to(device)/255, data[1].to(device), data[2]
            bs, num_frames = inputs.shape[0], inputs.shape[1]
            reshaped_inputs = torch.flatten(inputs, start_dim = 0, end_dim = 1)
            input_frames = reshaped_inputs.permute(0, 4, 1, 2, 3)
            features, _, _ = model(input_frames.float())
            features = features.reshape((bs, num_frames, features.shape[-1]))
            features_video = features.cpu().detach().numpy()
            for batch_ind in range(bs):
                name = videos[batch_ind]
                np.save(os.path.join('split_' + str(ind), "features_timesformer_all", name.replace("mp4", "npy")), features_video[batch_ind])
        for i, data in enumerate(tqdm(val_loader)):
            inputs, labels, videos = data[0].to(device)/255, data[1].to(device), data[2]
            bs, num_frames = inputs.shape[0], inputs.shape[1]
            reshaped_inputs = torch.flatten(inputs, start_dim = 0, end_dim = 1)
            input_frames = reshaped_inputs.permute(0, 4, 1, 2, 3)
            features, _, _ = model(input_frames.float())
            features = features.reshape((bs, num_frames, features.shape[-1]))
            features_video = features.cpu().detach().numpy()
            for batch_ind in range(bs):
                name = videos[batch_ind]
                np.save(os.path.join('split_' + str(ind), "features_timesformer_all", name.replace("mp4", "npy")), features_video[batch_ind])
        for i, data in enumerate(tqdm(test_loader)):
            inputs, labels, videos = data[0].to(device)/255, data[1].to(device), data[2]
            bs, num_frames = inputs.shape[0], inputs.shape[1]
            reshaped_inputs = torch.flatten(inputs, start_dim = 0, end_dim = 1)
            input_frames = reshaped_inputs.permute(0, 4, 1, 2, 3)
            features, _, _ = model(input_frames.float())
            features = features.reshape((bs, num_frames, features.shape[-1]))
            features_video = features.cpu().detach().numpy()
            for batch_ind in range(bs):
                name = videos[batch_ind]
                np.save(os.path.join('split_' + str(ind), "features_timesformer_all", name.replace("mp4", "npy")), features_video[batch_ind])
                
if __name__ == "__main__":
    get_features(9)
