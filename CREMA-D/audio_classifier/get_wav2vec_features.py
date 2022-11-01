import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
import librosa
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import torch.nn as nn
import random
from sklearn.metrics import f1_score
from tqdm import tqdm
from train_ft_wav2vec import MyDataset, WAV2VECGRUSentiment, create_dataset

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

def get_features(ind):
    train_loader = create_dataset("train", ind)
    val_loader = create_dataset("val", ind)
    test_loader = create_dataset("test", ind)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
    wav2vec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h", output_hidden_states=True)

    model = WAV2VECGRUSentiment(wav2vec, 100, 6, 2, True, 0.5)
    checkpoint = torch.load(os.path.join('..', 'split_' + str(ind), 'best_model_wav2vec_ft' + str(ind) + '.tar'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    if os.path.exists(os.path.join('..', 'split_' + str(ind), "features_wav2vec")) == False:
        os.mkdir(os.path.join("..", 'split_' + str(ind), "features_wav2vec"))
    with torch.no_grad():
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels, videos = data[0].to(device), data[1].to(device), data[2]
            inputs = processor(inputs, sampling_rate=16000, return_tensors="pt")
            _, _, seq = model(inputs['input_values'].to(device))
            for vid_ind in range(len(videos)):
                name = videos[vid_ind]
                features_video = seq[vid_ind, :, :].cpu().detach().numpy()
                np.save(os.path.join("..", 'split_' + str(ind), "features_wav2vec", name.replace("wav", "npy")), features_video)
        for i, data in enumerate(tqdm(val_loader)):
            inputs, labels, videos = data[0].to(device), data[1].to(device), data[2]
            inputs = processor(inputs, sampling_rate=16000, return_tensors="pt")
            _, _, seq = model(inputs['input_values'].to(device))
            for vid_ind in range(len(videos)):
                name = videos[vid_ind]
                features_video = seq[vid_ind, :, :].cpu().detach().numpy()
                np.save(os.path.join("..", 'split_' + str(ind), "features_wav2vec", name.replace("wav", "npy")), features_video)
        for i, data in enumerate(tqdm(test_loader)):
            inputs, labels, videos = data[0].to(device), data[1].to(device), data[2]
            inputs = processor(inputs, sampling_rate=16000, return_tensors="pt")
            _, _, seq = model(inputs['input_values'].to(device))
            for vid_ind in range(len(videos)):
                name = videos[vid_ind]
                features_video = seq[vid_ind, :, :].cpu().detach().numpy()
                np.save(os.path.join("..", 'split_' + str(ind), "features_wav2vec", name.replace("wav", "npy")), features_video)
                
if __name__ == "__main__":
    get_features(5)
