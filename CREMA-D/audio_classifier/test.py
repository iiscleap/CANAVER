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


#Logger set
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
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
    wav2vec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h", output_hidden_states=True)

    model = WAV2VECGRUSentiment(wav2vec, 100, 6, 2, True, 0.5)
    checkpoint = torch.load('best_model_wav2vec_ft' + str(ind) + '.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    with torch.no_grad():
        pred_test = []
        gt_test = []
        for i, data in enumerate(tqdm(test_loader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = processor(inputs, sampling_rate=16000, return_tensors="pt")
            test_out, _, _ = model(inputs['input_values'].to(device))
            pred = torch.argmax(test_out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_test.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_test.extend(labels)

    test_f1 = f1_score(gt_test, pred_test, average='weighted')
    logger.info(f"Test Accuracy {test_f1}")

if __name__ == "__main__":
    test(1)