# Voice Enhancement AI

A simple project for reducing background noise and improving voice clarity in noisy audio.
The system combines a spectral denoiser with a lightweight CNN model and includes a Streamlit demo app.

Live Demo  : https://speech-enhance-ai.streamlit.app/

## Project Structure

- `app/` - Streamlit demo app for audio upload and enhancement
- `data/` - raw audio, noisy/clean file pairs, and metadata
- `models/` - saved model checkpoints
- `src/` - core source code for data loading, denoising, model training, and inference
- `tests/` - unit tests for denoising and inference

## Key Files

- `src/processing/denoise.py` - spectral subtraction denoiser
- `src/models/cnn_model.py` - lightweight CNN used for audio enhancement
- `src/inference/enhance_audio.py` - enhancement pipeline used by the app
- `src/training/train.py` - training script for the CNN model
- `app/streamlit_app.py` - web UI for uploading noisy audio and downloading enhanced output

## Models

- `models/best_model.pth` - best validation checkpoint saved during training
- `models/final_model.pth` - final checkpoint saved after training completes

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run the training script to train the CNN model:

```bash
python src/training/train.py --epochs 15 --batch_size 16 --lr 0.001 --duration 1.0
```

This uses `data/metadata/file_pairs.csv` for noisy/clean audio pairs. If the metadata file is missing, the training script will try to generate it automatically.

## Inference

The enhancement workflow is:

1. Load noisy audio
2. Apply spectral denoising
3. Optionally run the trained CNN model for further enhancement
4. Save the enhanced audio file

The Streamlit app and `src/inference/enhance_audio.py` both use this flow.

## Run the Streamlit App

Start the demo with:

```bash
streamlit run app/streamlit_app.py
```

Then upload a noisy audio file, choose an enhancement mode, and click **Enhance audio**.

## Notes

- The default enhancement path uses the spectral denoiser.
- The CNN model is a small 1D conv network trained to improve noisy speech.
- If `models/best_model.pth` is available, the app will use it for the deep learning option.

