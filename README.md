# Audio Deepfake Detection using LCNN

This project aims to detect audio deepfakes or spoofed audio using a Lightweight Convolutional Neural Network (LCNN) combined with a Gated Recurrent Unit (GRU) for capturing temporal dependencies. The model is designed for low-latency applications and achieves an Equal Error Rate (EER) of approximately 0.8%.

## Dataset
The dataset used for this project is the [ASVspoof 2019 dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset), available on Kaggle.

### Dataset Restructuring
The original structure of the dataset was modified to facilitate training and evaluation:

1. **Protocol Files Consolidation:**
   - The protocol files from the original dataset were consolidated into a single format:
     - `ASVspoof 2019 dataset/protocols/ASVspoof2019.LA.{split}.txt`
     - Each protocol file (for `train`, `dev`, `eval`) was reformatted to have three space-separated columns:
       - `speaker_id`, `file_name`, `label` (bonafide or spoof)
   - These consolidated files were later used to create dataframes.

2. **File Renaming:**
   - Audio files were renamed and moved to a consolidated directory structure:
     - `ASVspoof 2019 dataset/audio/{split}/flac/LA_{split}_*.flac`
   - All audio files from each split were gathered into a unified `audio/` folder.

3. **Dataframe Enhancement:**
   - For each split, a new feature named `audio_path` was created to map each row to the absolute path of its corresponding `.flac` file.
   - Using this path, a spectrogram was generated for each file, and LFCC (Linear Frequency Cepstral Coefficients) features were extracted.

## Model Architecture
The model consists of an LCNN for spatial pattern recognition, complemented by a GRU to capture temporal dependencies. The extracted LFCC features are fed into this network to produce the final classification.

### Training Details
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy Loss
- **Performance:** Achieved an EER of approximately 0.8%

## Installation and Usage
### Modules Used
- librosa
- torch
- pandas
- numpy
- tqdm
- sklearn.metrics
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/audio-deepfake-detection.git
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

