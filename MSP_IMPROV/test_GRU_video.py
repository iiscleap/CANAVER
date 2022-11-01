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
from video_GRU import MyDataset, create_dataset, GRUModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--SEED', type=int, required=True)
args = parser.parse_args()

SEED = args.SEED 

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

def test(ind):
    test_loader = create_dataset("test", ind)
    model = GRUModel(768, 384, 4, 0.5, 2, True)
    checkpoint = torch.load('context_video.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    pred_test, gt_test = [], []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            video_features, labels, name = data[0].to(device), data[1].to(device), data[2]
            _, pred = model(video_features)
            pred_class = torch.argmax(pred, dim = 1)
            pred_class = pred_class.detach().cpu().numpy()
            pred_class = list(pred_class)
            pred_test.extend(pred_class)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_test.extend(labels)
    test_f1 = f1_score(gt_test, pred_test, average='weighted')
    logger.info(f"Test Accuracy {test_f1}")

if __name__ == "__main__":
    test(6)
