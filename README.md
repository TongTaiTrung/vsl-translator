# VSL Translator

Vietnamese Sign Language (VSL) translator using **MediaPipe Holistic** for landmark extraction and an **LSTM** classifier (Keras/TensorFlow) for sign recognition.

## Project Structure

```
vsl-translator/
├── config.py                  # Central configuration (paths, actions, hyperparams)
├── src/vsl_translator/        # Core library
│   ├── __init__.py
│   ├── dataset.py             # Keypoint sequence dataset loader
│   ├── landmarks.py           # MediaPipe holistic detection & keypoint extraction
│   └── model.py               # LSTM model definition
├── scripts/                   # Runnable CLI scripts
│   ├── collect_data.py        # Collect keypoint data from webcam
│   ├── train.py               # Train the LSTM classifier
│   ├── predict.py             # Run predictions on the test set
│   ├── cam_test.py            # Live webcam landmark visualisation
│   └── inspect_videos.py      # Inspect a video dataset
├── data/                      # Collected keypoint sequences (.npy)
├── models/                    # Trained model weights (.h5)
├── logs/                      # TensorBoard training logs
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Test webcam + MediaPipe
python scripts/cam_test.py

# 2. Collect training data (e.g. for "hello", "thanks", "iloveyou")
python scripts/collect_data.py --action hello
python scripts/collect_data.py --action thanks
python scripts/collect_data.py --action iloveyou

# 3. Train the model (saves held-out test set to models/test_set.npz)
python scripts/train.py

# 4. Evaluate on the held-out test set
python scripts/predict.py

# 5. Live webcam inference with temporal smoothing
python scripts/predict_live.py
```
