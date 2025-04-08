import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

class EmotionDetector:
    def __init__(self, ravdess_dir="ravdess"):
        self.ravdess_dir = ravdess_dir
        self.emotion_map = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        self.sample_features = {}
        self.load_ravdess_samples()
        
    def load_ravdess_samples(self):
        """Load RAVDESS audio files and extract their features."""
        if not os.path.exists(self.ravdess_dir):
            print(f"RAVDESS directory not found. Creating directory: {self.ravdess_dir}")
            os.makedirs(self.ravdess_dir)
            print("\nPlease download and extract the RAVDESS dataset into this folder:")
            print("https://zenodo.org/record/1188976")
            return
            
        actor_dirs = [d for d in os.listdir(self.ravdess_dir) 
                      if os.path.isdir(os.path.join(self.ravdess_dir, d)) 
                      and d.startswith('Actor_')]
        
        if not actor_dirs:
            print(f"No actor directories found in {self.ravdess_dir}")
            return
        
        print(f"Found {len(actor_dirs)} actor directories. Processing...")
        total_files = 0
        
        for actor_dir in actor_dirs:
            actor_path = os.path.join(self.ravdess_dir, actor_dir)
            audio_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
            total_files += len(audio_files)
            
            for file in audio_files:
                try:
                    parts = file.split('-')
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        emotion = self.emotion_map.get(emotion_code)
                        
                        if emotion:
                            if emotion not in self.sample_features:
                                self.sample_features[emotion] = []
                                
                            file_path = os.path.join(actor_path, file)
                            features = self.extract_features(file_path)
                            if features is not None:
                                self.sample_features[emotion].append(features)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    
        if self.sample_features:
            print(f"\nLoaded {total_files} files.")
            for emotion, samples in self.sample_features.items():
                print(f"{emotion}: {len(samples)} samples")
        else:
            print("No valid samples loaded.")

    def extract_features(self, audio_path, n_mfcc=13):
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            y, _ = librosa.effects.trim(y)  # Trim silence
            y = librosa.util.normalize(y)   # Normalize
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            features = np.vstack([mfcc, chroma, contrast])
            features = (features - np.mean(features)) / np.std(features)
            return features.T
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None

    def record_audio(self, duration=5, sample_rate=22050):
        print(f"\nRecording for {duration} seconds...")
        recording = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1)
        sd.wait()
        print("Recording complete.")
        return recording.flatten()

    def detect_emotion(self, audio_data):
        if not self.sample_features:
            print("No features loaded.")
            return None, None
        
        y = librosa.util.normalize(audio_data)
        y, _ = librosa.effects.trim(y)
        mfcc = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=22050)
        contrast = librosa.feature.spectral_contrast(y=y, sr=22050)
        features = np.vstack([mfcc, chroma, contrast])
        features = (features - np.mean(features)) / np.std(features)
        features = features.T
        
        min_distance = float('inf')
        detected_emotion = None
        distances = {}

        for emotion, samples in self.sample_features.items():
            emotion_distances = []
            for sample in samples:
                distance, _ = fastdtw(features, sample, dist=euclidean)
                emotion_distances.append(distance)
            top_k = sorted(emotion_distances)[:3]  # Use top 3 matches
            avg_distance = np.mean(top_k)
            distances[emotion] = avg_distance
            if avg_distance < min_distance:
                min_distance = avg_distance
                detected_emotion = emotion
        
        total = sum(distances.values())
        confidence = 1 - (min_distance / total) if total > 0 else 0
        return detected_emotion, confidence

    def plot_mfcc(self, mfcc, title="MFCC Features"):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()

def main():
    detector = EmotionDetector()
    
    while True:
        print("\n1. Record and detect emotion")
        print("2. Exit")
        choice = input("Enter your choice (1/2): ")
        
        if choice == '1':
            if not detector.sample_features:
                print("\nSamples not loaded properly. Check dataset.")
                continue
                
            audio_data = detector.record_audio(duration=5)
            emotion, confidence = detector.detect_emotion(audio_data)
            
            if emotion:
                print(f"\nDetected Emotion: {emotion}")
                print(f"Confidence: {confidence:.2%}")
                
                mfcc = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=13)
                detector.plot_mfcc(mfcc, title=f"MFCC - Detected: {emotion}")
            else:
                print("Could not detect emotion. Try again.")
        
        elif choice == '2':
            print("Exiting...")
            break
        else:
            print("Invalid input.")

if __name__ == "__main__":
    main()
