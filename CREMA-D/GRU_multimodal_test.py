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
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from GRU_multimodal_classifier import MyDataset, MultimodalClassifier, create_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
np.random.seed(1234)
torch.manual_seed(1234)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def test(ind):
    test_loader = create_dataset("test", ind)
    model = MultimodalClassifier(1568, 249, 768, 2)
    checkpoint = torch.load('seq_multimodal_model' + str(ind) + '.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    pred_test = []
    gt_test = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            video_features, wav_features, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            # vid_cls = torch.zeros(len(labels), 1, 768).to(device)
            # aud_cls = torch.zeros(len(labels), 1, 768).to(device)
            # video_features = torch.cat((vid_cls, video_features), 1)
            # wav_features = torch.cat((aud_cls, wav_features), 1)
            pred = model(video_features, wav_features.float())
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
    test(2)
