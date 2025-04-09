# Emotion Detection from Speech

## 1. Analysis of Challenges in Speech Emotion Detection

The challenges are listed below:

- **Feature Extraction Complexity**: Identifying which acoustic features best represent emotional content is challenging.

- **Individual Variability**: Different people express emotions differently based on cultural background, personality, and speaking style. The same emotion can have different acoustic manifestations across speakers.

- **Context Dependency**: Emotions in speech are often context-dependent, making them hard to identify in isolation.

- **Class Imbalance**: The RAVDESS dataset may have uneven distribution of emotion samples, which can bias the model toward more represented emotions.

- **Temporal Dynamics**: Emotions evolve over time in speech, requiring algorithms that can handle variable-length sequences. The DTW algorithm addresses this but adds computational complexity.

- **Low-Resource Setting**: Speech emotion detection typically requires substantial labeled data, which can be expensive and time-consuming to collect.


## 2. Application Selection

Implemented a **real-time speech emotion recognition system** using traditional signal processing techniques. This application is appropriate for:

- Tracking emotional states over time for therapeutic purposes
- Evaluating customer satisfaction in call centers
- Enabling more empathetic responses from virtual assistants
- Assessing student engagement and emotional responses during online learning

The choice of DTW with acoustic features is suitable for a situation where:
- Training data is limited (as DTW is instance-based and doesn't require extensive training)
- Real-time processing is needed

## 3. Architecture 

![alt text](image.png)

## 4. Module Description

### 4.1 Data Collection Module
- **Purpose**: Acquires audio data for emotion analysis
- **Components**:
  - **RAVDESS Dataset Loader**: Loads pre-recorded emotional speech samples
  - **Audio Recorder**: Captures real-time speech for emotion detection
- **Techniques Used**:
  - Uses `sounddevice` for real-time audio capture at 22.05 kHz sampling rate
  - Directory traversal for organized dataset loading
  - Audio file metadata parsing to extract emotion labels

### 4.2 Feature Extraction Module
- **Purpose**: Transforms raw audio into representative features
- **Components**:
  - **MFCC Extractor**: Captures vocal tract configuration information
  - **Chroma Feature Extractor**: Represents tonal content
  - **Spectral Contrast Extractor**: Captures distribution of sound energy
- **Techniques Used**:
  - Librosa for audio processing and feature extraction
  - 13 MFCCs to represent speech spectral envelope
  - Stacked features to capture complementary acoustic information

### 4.3 Preprocessing Module
- **Purpose**: Enhances signal quality and standardizes features
- **Components**:
  - **Silence Trimmer**: Removes non-speech segments
  - **Normalizer**: Adjusts amplitude
  - **Feature Standardizer**: Z-score normalization
- **Techniques Used**:
  - Librosa's `effects.trim()` to remove silence
  - Standard normalization to unit variance using NumPy
  - Feature stacking to combine different feature types

### 4.4 Feature Database Module
- **Purpose**: Stores reference features for emotion classification
- **Components**:
  - **Emotion Template Storage**: Organizes feature vectors by emotion
  - **Sample Management**: Handles multiple samples per emotion
- **Techniques Used**:
  - Dictionary-based storage for easy emotion-based retrieval
  - Multi-sample representation to capture variation within emotions

### 4.5 Emotion Classification Module
- **Purpose**: Compares input features to database to identify emotions
- **Components**:
  - **DTW Distance Calculator**: Measures similarity between sequences
  - **K-Nearest Averaging**: Reduces outlier impact
  - **Confidence Calculator**: Estimates reliability of classification
- **Techniques Used**:
  - Fast DTW algorithm to handle temporal variations efficiently
  - Euclidean distance metric for feature comparison
  - Top-3 averaging to improve robustness to outliers

### 4.6 Visualization Module
- **Purpose**: Provides visual feedback on the audio features
- **Components**:
  - **MFCC Plotter**: Creates spectrogram-like visualizations
- **Techniques Used**:
  - Matplotlib for visualization
  - Librosa's display functions for audio feature visualization

## 5. Data Selection and Preprocessing

### 5.1 Dataset Description
The system uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which contains:
- 24 professional actors (12 male, 12 female)
- 8 emotional expressions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Two intensity levels (normal and strong)
- Multiple repetitions of each emotion

### 5.2 Data Preprocessing Pipeline
1. **Audio Loading**: Audio files are loaded at 22.05 kHz sample rate
2. **Silence Removal**: Leading and trailing silence are trimmed
3. **Amplitude Normalization**: Audio is normalized to have consistent volume
4. **Feature Extraction**:
   - 13 MFCCs are extracted to capture vocal tract configuration
   - Chroma features represent tonal content with 12 dimensions
   - Spectral contrast captures energy distribution
5. **Feature Standardization**: Z-score normalization (mean=0, std=1)
6. **Feature Stacking**: All features are combined into a single matrix
7. **Transposition**: Features are transposed for DTW compatibility

### 5.3 Implementation Logic

The implementation follows this sequence:

1. Initialize the EmotionDetector class
2. Load and preprocess the RAVDESS dataset
3. Extract and store feature templates for each emotion
4. When detecting emotions:
   - Record audio input
   - Process the audio through the same feature extraction pipeline
   - Compare features to stored templates using DTW
   - Select the emotion with minimum average distance
   - Calculate confidence based on relative distances

### 5.4 Algorithm Implementation

The core algorithm is based on DTW (Dynamic Time Warping), which:
- Aligns two time series by warping the time axis
- Handles variations in speaking rate and duration
- Computes a distance measure that represents similarity

For each input:
1. The system calculates DTW distances to all reference samples
2. For each emotion category, it selects the top-3 closest matches
3. It averages these distances to get a per-emotion score
4. The emotion with the lowest average distance is selected
5. Confidence is calculated as a function of relative distances

## 7. Conclusion and Future Work

### 7.1 Conclusion

This speech emotion detection system demonstrates a practical approach to emotion recognition using acoustic features and DTW. The architecture balances complexity with performance by using established signal processing techniques rather than deep learning approaches. This makes it suitable for environments with limited computational resources or training data.

The system's strengths lie in its intuitive approach, explainability, and ability to work with limited samples. However, its performance may be limited by the computational complexity of DTW and its dependency on acoustic features without linguistic context.

### 7.2 Future Work

Several enhancements could improve the system:

1. **Integration of Deep Learning**: Replace or augment DTW with neural network models
   - Implement a CNN-LSTM architecture for sequential modeling
   - Use pre-trained audio embeddings like wav2vec

2. **Feature Enhancement**:
   - Add prosodic features (pitch contour, speaking rate)
   - Include voice quality parameters (jitter, shimmer)
   - Explore wavelet-based features for better time-frequency resolution
