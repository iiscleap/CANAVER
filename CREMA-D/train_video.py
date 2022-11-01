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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

class MyDataset(Dataset):
    '''Samples 8 frames uniformly at random from the entire video duration.
    Returns the frames, class label and name of the video file
    '''
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
        num_frames = 8
        if frames.shape[0] > num_frames:
            start = np.random.randint(0, frames.shape[0], num_frames)
            start.sort()
            start = list(start)
            final_sig = frames[start]
        else:
            pad_begin_len = random.randint(0, num_frames - frames.shape[0])
            pad_end_len = num_frames - frames.shape[0] - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((pad_begin_len, frames.shape[1], frames.shape[2], frames.shape[3]))
            pad_end = torch.zeros((pad_end_len, frames.shape[1], frames.shape[2], frames.shape[3]))
            final_sig = torch.cat((pad_begin, frames, pad_end), 0)

        return final_sig, class_id, self.vid_files[vid_ind]

def create_dataset(mode, ind):
    if mode == 'train':
        folder = "mp4"
        f = open("splits/train_labels" + str(ind) + ".json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "mp4"
        f = open("splits/val_labels" + str(ind) + ".json")
        labels = json.load(f)
        f.close()
    else:
        folder = "mp4"
        f = open("splits/test_labels" + str(ind) + ".json")
        labels = json.load(f)
        f.close()
    dataset = MyDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=4,
                    shuffle=True,
                    drop_last=False)
    return loader

def compute_loss(output, labels):
    ce_loss = nn.CrossEntropyLoss(reduction='none')(output, labels.long())
    pt = torch.exp(-ce_loss)
    loss = ((1-pt)**0 * ce_loss).mean()
    return loss

def train(ind):
    '''Loads the timesformer model pre-trained on Kinetics-600 dataset.
    Trains and saves the fine-tuned model for feature extraction.
    '''
    train_loader = create_dataset("train", ind)
    val_loader = create_dataset("val", ind)

    model = TimeSformer(img_size=224, num_classes=6,
                        num_frames=8, attention_type='divided_space_time',
                        pretrained_model='../../TimeSformer/TimeSformer_divST_8x32_224_K600.pyth')
    model.to(device)

    
    base_lr = 5e-5
    optimizer = Adam([{'params':model.parameters(), 'lr':base_lr}])
    final_val_loss = 99999

    for e in range(10):
        model.train()
        tot_loss, tot_correct = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0
        val_correct = 0.0
        train_size = 0
        val_size = 0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        #handle = model.VisionTransformer.register_forward_hook(get_activation('feat'))
        
        for i, data in enumerate(tqdm(train_loader)):
            model.zero_grad()
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device)/255, data[1].to(device)
            input_frame = inputs.permute(0, 4, 1, 2, 3)
            feats, seq, out = model(input_frame.float())
            loss = compute_loss(out, labels)
            tot_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                inputs, labels, name = data[0].to(device)/255, data[1].to(device), data[2]
                input_frame = inputs.permute(0, 4, 1, 2, 3)
                _, _, out = model(input_frame.float())
                loss = compute_loss(out, labels)
                val_loss += loss.item()
                pred = torch.argmax(out, dim = 1)
                pred = pred.detach().cpu().numpy()
                pred = list(pred)
                pred_val.extend(pred)
                labels = labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val.extend(labels)
        if val_loss < final_val_loss:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),},
                        'video_timesformer_pretrained_' + str(ind) + '.tar')
            final_val_loss = val_loss
        train_loss = tot_loss/len(train_loader)
        train_f1 = f1_score(gt_tr, pred_tr, average='weighted')
        #train_acc = tot_correct/train_size
        val_loss_log = val_loss/len(val_loader)
        val_f1 = f1_score(gt_val, pred_val, average='weighted')
        #val_acc_log = val_correct/val_size
        e_log = e + 1
        logger.info(f"Epoch {e_log}, \
                    Training Loss {train_loss},\
                    Training Accuracy {train_f1}")
        logger.info(f"Epoch {e_log}, \
                    Validation Loss {val_loss_log},\
                    Validation Accuracy {val_f1}")

if __name__ == "__main__":
    train(1)
