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
from sklearn.metrics import f1_score
import gc
from tqdm import tqdm 
from video_GRU import MyDataset, GRUModel, create_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_features(ind):
    train_loader = create_dataset("train", ind)
    val_loader = create_dataset("val", ind)
    test_loader = create_dataset("test", ind)
    model = GRUModel(768, 384, 6, 0.5, 2, True)
    checkpoint = torch.load('context_video' + str(ind) + '.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    if os.path.exists(os.path.join('split_' + str(ind), "features_video_GRU")) == False:
        os.mkdir(os.path.join('split_' + str(ind), "features_video_GRU"))
    with torch.no_grad():
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels, videos = data[0].to(device), data[1].to(device), data[2]
            bs, num_frames = inputs.shape[0], inputs.shape[1]
            features, _ = model(inputs)
            features_video = features.cpu().detach().numpy()
            for batch_ind in range(bs):
                name = videos[batch_ind]
                np.save(os.path.join('split_' + str(ind), "features_video_GRU", name.replace("mp4", "npy")), features_video[batch_ind])
        for i, data in enumerate(tqdm(val_loader)):
            inputs, labels, videos = data[0].to(device), data[1].to(device), data[2]
            bs, num_frames = inputs.shape[0], inputs.shape[1]
            features, _ = model(inputs)
            features_video = features.cpu().detach().numpy()
            for batch_ind in range(bs):
                name = videos[batch_ind]
                np.save(os.path.join('split_' + str(ind), "features_video_GRU", name.replace("mp4", "npy")), features_video[batch_ind])
        for i, data in enumerate(tqdm(test_loader)):
            inputs, labels, videos = data[0].to(device), data[1].to(device), data[2]
            bs, num_frames = inputs.shape[0], inputs.shape[1]
            features, _ = model(inputs)
            features_video = features.cpu().detach().numpy()
            for batch_ind in range(bs):
                name = videos[batch_ind]
                np.save(os.path.join('split_' + str(ind), "features_video_GRU", name.replace("mp4", "npy")), features_video[batch_ind])

if __name__ == "__main__":
    get_features(2)
