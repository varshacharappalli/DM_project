# Real-time Emotion Detection using DTW and MFCC

This project implements a real-time emotion detection system that uses Dynamic Time Warping (DTW) and Mel-Frequency Cepstral Coefficients (MFCC) to detect emotions in recorded audio. The system compares the recorded audio with sample audio files to determine the closest matching emotion.

## Features

- Real-time audio recording using sounddevice
- MFCC feature extraction
- Dynamic Time Warping for emotion detection
- Visualization of MFCC features
- Support for multiple emotion categories

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Project Structure

- `emotion_detection.py`: Main script for emotion detection
- `requirements.txt`: List of required Python packages
- `samples/`: Directory for sample audio files (created automatically)

## Setup

1. Create a directory structure for sample audio files:
```
samples/
├── happy/
│   ├── sample1.wav
│   └── sample2.wav
└── sad/
    ├── sample1.wav
    └── sample2.wav
```

2. Add sample audio files for each emotion category in their respective directories.

## Usage

1. Run the emotion detection script:
```bash
python emotion_detection.py
```

2. Choose option 1 to record and detect emotion:
   - The system will record audio for 5 seconds
   - It will extract MFCC features from the recorded audio
   - Compare with sample audio files using DTW
   - Display the detected emotion and DTW distance
   - Show a visualization of the MFCC features

## How It Works

1. **Sample Loading**:
   - The system loads sample audio files for each emotion category
   - Extracts and normalizes MFCC features from each sample

2. **Audio Recording**:
   - Records audio from the microphone for a specified duration
   - Uses sounddevice for real-time recording

3. **Feature Extraction**:
   - Extracts MFCC features from the recorded audio
   - Normalizes the features for comparison

4. **Emotion Detection**:
   - Uses Dynamic Time Warping to compare the recorded audio with samples
   - Determines the closest matching emotion based on DTW distance

5. **Visualization**:
   - Displays the MFCC features of the recorded audio
   - Shows the detected emotion and confidence score

## Notes

- Ensure you have a working microphone
- Sample audio files should be in WAV format
- The system supports multiple emotion categories (add more directories in the samples folder)
- DTW distance indicates how close the match is (lower distance means better match) 