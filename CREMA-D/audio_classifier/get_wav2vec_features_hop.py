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
from train_ft_wav2vec import WAV2VECGRUSentiment

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
        wav_files = list(target_dict.keys())
        wav_files = [x.replace("flv", "wav") for x in wav_files]
        self.wav_files = wav_files
        self.sr = 16000
        self.duration = 5000

    def __len__(self):
        return len(self.wav_files) 

    def __getitem__(self, audio_ind):
        audio_file = os.path.join(self.folder, self.wav_files[audio_ind])
        class_id = self.target[self.wav_files[audio_ind].replace("wav", "flv")]
        (sig, sr) = librosa.load(audio_file)

        aud = (sig, sr)
        reaud = (sig, self.sr)
        resig = sig
        sig_len = resig.shape[0]
        max_len = self.sr//1000 * self.duration
        tot_frames = max_len
        frame_window, frame_hop = 1200*16, 600*16
        frame_list = []
        if len(resig.shape) == 2:
            resig = np.mean(resig, axis = 1)

        if (sig_len > max_len):
            # Truncate the signal to the given length
            start = np.random.randint(0, sig_len-max_len)

            final_sig = resig[start:start+max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = np.zeros((pad_begin_len))
            pad_end = np.zeros((pad_end_len))

            final_sig = np.float32(np.concatenate((pad_begin, resig, pad_end), 0))
            final_aud = (final_sig, self.sr)
        
        seq_length = max_len//frame_hop + 1
        for ind in range(seq_length):
            start_index = ind*frame_window
            end_index = start_index + frame_hop + 1
            end_index = min(tot_frames, end_index)
            frame_windowed = torch.Tensor(final_sig[start_index:end_index])
            if frame_windowed.shape[0] < frame_window:
                padding = torch.zeros((frame_window-frame_windowed.shape[0]))
                frame_windowed = torch.cat((frame_windowed, padding), 0)
                frame_list.append(frame_windowed)
        audio_sig = torch.cat([x.float() for x in frame_list])
        audio_sig = audio_sig.reshape((len(frame_list), frame_window))

        return audio_sig, class_id, self.wav_files[audio_ind]

def create_dataset(mode, ind):
    folder = "wav_files"
    if mode == 'train':
        f = open("../splits/train_labels" + str(ind) + ".json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        f = open("../splits/val_labels" + str(ind) + ".json")
        labels = json.load(f)
        f.close()
    else:
        f = open("../splits/test_labels" + str(ind) + ".json")
        labels = json.load(f)
        f.close()
    dataset = MyDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=16,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False)
    return loader


def get_features(ind):
    train_loader = create_dataset("train", ind)
    val_loader = create_dataset("val", ind)
    test_loader = create_dataset("test", ind)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
    wav2vec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h", output_hidden_states=True)

    model = WAV2VECGRUSentiment(wav2vec, 100, 6, 2, True, 0.5)
    checkpoint = torch.load(os.path.join('..', 'split_' + str(ind), '1200best_model_wav2vec_ft' + str(ind) + '.tar'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    if os.path.exists(os.path.join('..', 'split_' +str(ind), "features_wav2vec_new")) == False:
        os.mkdir(os.path.join('..', 'split_' +str(ind), "features_wav2vec_new"))
    with torch.no_grad():
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels, names = data[0].to(device), data[1].to(device), data[2]
            bs, num_frames = inputs.shape[0], inputs.shape[1]
            reshaped_inputs = torch.flatten(inputs, start_dim = 0, end_dim = 1)
            reshaped_inputs = processor(reshaped_inputs, sampling_rate=16000, return_tensors="pt")
            _, features, _ = model(reshaped_inputs['input_values'].to(device))
            features = features.reshape((bs, num_frames, features.shape[-1]))
            features_audio = features.cpu().detach().numpy()
            for batch_ind in range(bs):
                name = names[batch_ind]
                np.save(os.path.join("..", 'split_' +str(ind), "features_wav2vec_new", name.replace("wav", "npy")), features_audio[batch_ind])
        for i, data in enumerate(tqdm(val_loader)):
            inputs, labels, names = data[0].to(device), data[1].to(device), data[2]
            bs, num_frames = inputs.shape[0], inputs.shape[1]
            reshaped_inputs = torch.flatten(inputs, start_dim = 0, end_dim = 1)
            reshaped_inputs = processor(reshaped_inputs, sampling_rate=16000, return_tensors="pt")
            _, features, _ = model(reshaped_inputs['input_values'].to(device))
            features = features.reshape((bs, num_frames, features.shape[-1]))
            features_audio = features.cpu().detach().numpy()
            for batch_ind in range(bs):
                name = names[batch_ind]
                np.save(os.path.join("..", 'split_' +str(ind), "features_wav2vec_new", name.replace("wav", "npy")), features_audio[batch_ind])
        for i, data in enumerate(tqdm(test_loader)):
            inputs, labels, names = data[0].to(device), data[1].to(device), data[2]
            bs, num_frames = inputs.shape[0], inputs.shape[1]
            reshaped_inputs = torch.flatten(inputs, start_dim = 0, end_dim = 1)
            reshaped_inputs = processor(reshaped_inputs, sampling_rate=16000, return_tensors="pt")
            _, features, _ = model(reshaped_inputs['input_values'].to(device))
            features = features.reshape((bs, num_frames, features.shape[-1]))
            features_audio = features.cpu().detach().numpy()
            for batch_ind in range(bs):
                name = names[batch_ind]
                np.save(os.path.join("..", 'split_' +str(ind), "features_wav2vec_new", name.replace("wav", "npy")), features_audio[batch_ind])
        
                
if __name__ == "__main__":
    get_features(2)
