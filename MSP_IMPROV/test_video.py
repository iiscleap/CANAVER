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
from train_video import MyDataset, create_dataset, TimesformerSentiment

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

def test(ind):
    test_loader = create_dataset("test", ind)
    timesformer = TimeSformer(img_size=224, num_classes=4,
                        num_frames=8, attention_type='divided_space_time',
                        pretrained_model='../../TimeSformer/TimeSformer_divST_8x32_224_K600.pyth')
    model = TimesformerSentiment(timesformer, 768, 4)
    checkpoint = torch.load('video_timesformer_pretrained.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    pred_test = []
    gt_test = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            inputs, labels = data[0].to(device)/255, data[1].to(device)
            input_frame = inputs.permute(0, 4, 1, 2, 3)
            _, _, out = model(input_frame.float())
            pred = torch.argmax(out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_test.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_test.extend(labels)

        test_f1 = f1_score(gt_test, pred_test, average='weighted')

        logger.info(f"Test Accuracy {test_f1}")

if __name__ == "__main__":
    test(12)
