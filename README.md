# CANAVER
Code for Multimodal Cross Attention Network for Audio Visual Emotion Recognition submiited to ICASSP 2023

Please download the video files of CREMA-D (https://github.com/CheyneyComputerScience/CREMA-D) and MSP-IMPROV (https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html) datasets. The files need to be converted to appropriate format, the helper code of which will be made available upon request.

The code available here can be considered as 4 distinct steps:

1. The audio classifier can be trained by running the file **train_ft_wav2vec.py** in the audio classifier folder. The windowed audio features as described in the paper can be extracted using the code **get_wav2vec_features_hop.py**. The feature extractor can only run once the model is trained.
2. The video feature extractor timesformer is to be trained using the repository (https://github.com/facebookresearch/TimeSformer) by running the file **train_video.py**. This is followed by the windowed feature extractor in the code **get_timesformer_features_hop.py**
3. Once the timesformer windowed features are available the GRU with self-attention can be trained by running the code **video_GRU.py**. The context enhanced GRU features for video is extracted by the code **vid_get_GRU_features.py**
4. Finally the multimodal block is trained using the code **GRU_multimodal_classifier.py**.
