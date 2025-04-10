import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from scipy.signal import stft
import time
import traceback

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
        self.sample_rate = 22050
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mfcc = 13
        self.sample_features = {}
        self.load_ravdess_samples()
        
    def load_ravdess_samples(self):
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
        processed_files = 0
        
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
                                processed_files += 1
                                if processed_files % 10 == 0:
                                    print(f"Processed {processed_files}/{total_files} files...")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    
        if self.sample_features:
            print(f"\nLoaded {processed_files} files successfully.")
            for emotion, samples in self.sample_features.items():
                print(f"{emotion}: {len(samples)} samples")
        else:
            print("No valid samples loaded.")

    def extract_features(self, audio_input, n_mfcc=13):
        try:
            if isinstance(audio_input, str):
                y, sr = librosa.load(audio_input, sr=22050)
            else:
                y = audio_input
                sr = 22050
                
            y, _ = librosa.effects.trim(y)
            y = librosa.util.normalize(y)
            
            # Extract MFCC features (more efficient)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            
            # Extracting only necessary features to improve performance
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            features = np.vstack([mfcc, chroma, contrast])
            # Normalize the features
            if np.std(features) != 0:  # Avoid division by zero
                features = (features - np.mean(features)) / np.std(features)
            return features.T
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            traceback.print_exc()
            return None

    def record_audio(self, duration=5, sample_rate=22050):
        print(f"\nRecording for {duration} seconds...")
        recording = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1)
        sd.wait()
        print("Recording complete.")
        return recording.flatten()

    def compute_dtw(self, seq1, seq2):
        if len(seq1) == 0 or len(seq2) == 0:
            return float('inf'), np.array([[0, 0]])
            
        n, m = len(seq1), len(seq2)
        
        # Use a reduced feature dimension for faster computation if sequences are large
        if n > 1000 or m > 1000:
            step_size = max(1, min(n, m) // 500)
            seq1 = seq1[::step_size]
            seq2 = seq2[::step_size]
            n, m = len(seq1), len(seq2)
        
        cost_matrix = np.full((n + 1, m + 1), np.inf)
        cost_matrix[0, 0] = 0
        
        path_matrix = np.zeros((n + 1, m + 1, 2), dtype=int)
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.sum((seq1[i-1] - seq2[j-1]) ** 2)
                
                min_cost = min(
                    cost_matrix[i-1, j],
                    cost_matrix[i, j-1],
                    cost_matrix[i-1, j-1]
                )
                
                cost_matrix[i, j] = cost + min_cost
                
                if min_cost == cost_matrix[i-1, j]:
                    path_matrix[i, j] = [i-1, j]
                elif min_cost == cost_matrix[i, j-1]:
                    path_matrix[i, j] = [i, j-1]
                else:
                    path_matrix[i, j] = [i-1, j-1]
        
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append([i-1, j-1])
            i, j = path_matrix[i, j]
        
        path.reverse()
        return cost_matrix[n, m], np.array(path)
    
    def detect_emotion(self, audio_data):
        if not self.sample_features:
            print("No sample features loaded. Please ensure RAVDESS dataset is available.")
            return None, None
            
        features = self.extract_features(audio_data)
        if features is None:
            print("Failed to extract features from audio data.")
            return None, None
            
        min_distance = float('inf')
        detected_emotion = None
        distances = {}
        
        print("Analyzing emotions...")
        total_samples = sum(len(samples) for samples in self.sample_features.values())
        processed = 0
        
        for emotion, samples in self.sample_features.items():
            distances[emotion] = []
            for sample in samples[:min(len(samples), 10)]:  # Limit number of samples to improve speed
                distance, _ = self.compute_dtw(features, sample)
                distances[emotion].append(distance)
                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed}/{total_samples} samples...")
                
            avg_distance = np.mean(distances[emotion]) if distances[emotion] else float('inf')
            if avg_distance < min_distance:
                min_distance = avg_distance
                detected_emotion = emotion
        
        # Calculate confidence
        if detected_emotion:
            avg_distances = [np.mean(dists) if dists else float('inf') for dists in distances.values()]
            avg_distances = [d for d in avg_distances if d != float('inf')]
            if avg_distances:
                total_distance = sum(avg_distances)
                confidence = 1 - (min_distance / total_distance) if total_distance else 0
            else:
                confidence = 0
        else:
            confidence = 0
                
        return detected_emotion, confidence
    
    # Add the missing plot_features method
    def plot_features(self, audio_data, title="Audio Features"):
        try:
            y = audio_data
            if isinstance(audio_data, str):
                y, sr = librosa.load(audio_data, sr=22050)
            sr = 22050
            
            plt.figure(figsize=(15, 10))
            
            # Plot waveform
            plt.subplot(3, 1, 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title('Waveform')
            
            # Plot MFCC
            plt.subplot(3, 1, 2)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            librosa.display.specshow(mfcc, x_axis='time', sr=sr)
            plt.colorbar(format='%+2.0f dB')
            plt.title('MFCC')
            
            # Plot spectrogram
            plt.subplot(3, 1, 3)
            spec = np.abs(librosa.stft(y))
            spec_db = librosa.amplitude_to_db(spec, ref=np.max)
            librosa.display.specshow(spec_db, y_axis='log', x_axis='time', sr=sr)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            
            plt.suptitle(title)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting features: {str(e)}")
            traceback.print_exc()
        
    def plot_dtw_path(self, seq1, seq2, path, title="DTW Path"):
        try:
            plt.figure(figsize=(10, 8))
            
            # Ensure seq1 and seq2 have at least one dimension for plotting
            if seq1.shape[1] > 0:
                feature_idx = 0
                plt.subplot(2, 1, 1)
                plt.plot(seq1[:, feature_idx], label='Sequence 1')
                plt.plot(seq2[:, feature_idx], label='Sequence 2')
                plt.title('Original Sequences (First Feature)')
                plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.imshow(np.zeros((len(seq1), len(seq2))), cmap='gray', aspect='auto')
            plt.plot(path[:, 1], path[:, 0], 'r-', linewidth=2)
            plt.title('DTW Warping Path')
            plt.xlabel('Sequence 2')
            plt.ylabel('Sequence 1')
            
            plt.suptitle(title)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting DTW path: {str(e)}")
            traceback.print_exc()

    def plot_mfcc(self, mfcc, title="MFCC Features"):
        try:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfcc, x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title(title)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting MFCC: {str(e)}")
            traceback.print_exc()

def main():
    detector = EmotionDetector()
    
    while True:
        print("\n1. Record and detect emotion")
        print("2. Exit")
        choice = input("Enter your choice (1/2): ")
        
        if choice == '1':
            if not detector.sample_features:
                print("\nNo samples loaded. Please ensure the RAVDESS dataset is properly set up.")
                continue
                
            try:
                audio_data = detector.record_audio(duration=5)
                
                start_time = time.time()
                print("Detecting emotion...")
                emotion, confidence = detector.detect_emotion(audio_data)
                end_time = time.time()
                print(f"Detection completed in {end_time - start_time:.2f} seconds")
                
                if emotion:
                    print(f"\nDetected emotion: {emotion}")
                    print(f"Confidence: {confidence:.2%}")
                    
                    # Plot the audio features
                    detector.plot_features(audio_data, title=f"Audio Features - Detected Emotion: {emotion}")
                    
                    # Extract features and find the closest sample
                    features = detector.extract_features(audio_data)
                    if features is not None and len(detector.sample_features.get(emotion, [])) > 0:
                        closest_sample = min(detector.sample_features[emotion], 
                                           key=lambda x: detector.compute_dtw(features, x)[0])
                        _, path = detector.compute_dtw(features, closest_sample)
                        detector.plot_dtw_path(features, closest_sample, path, 
                                             title=f"DTW Path - {emotion}")
                    else:
                        print("Cannot plot DTW path: Features extraction failed or no samples available.")
                else:
                    print("No emotion detected. Please ensure RAVDESS dataset is available.")
            except Exception as e:
                print(f"Error during emotion detection: {str(e)}")
                traceback.print_exc()
                
        elif choice == '2':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()