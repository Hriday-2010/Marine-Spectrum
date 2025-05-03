import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, 
                            confusion_matrix, classification_report)
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import pandas as pd
import warnings
import time
import soundfile as sf
import pickle
import hashlib
warnings.filterwarnings('ignore')

# --------------------------
# 1. Enhanced Data Loading & Preprocessing with Robust MP3 Support
# --------------------------

class AudioDataLoader:
    """Handles loading and preprocessing of audio datasets with better MP3 support"""
    
    def __init__(self, target_sr=22050, max_duration=5, n_mfcc=40, cache_dir='./cache'):
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.n_mfcc = n_mfcc
        self.le = LabelEncoder()
        self.supported_formats = ('.wav', '.mp3', '.WAV', '.MP3')  # Removed MP4 for simplicity
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.use_librosa_for_mp3 = True  # Try librosa first for MP3
        
    def _get_cache_key(self, root_path):
        """Generate a unique cache key based on dataset path and parameters"""
        params_hash = hashlib.md5(f"{self.target_sr}_{self.max_duration}_{self.n_mfcc}".encode()).hexdigest()
        path_hash = hashlib.md5(root_path.encode()).hexdigest()
        return f"{path_hash}_{params_hash}.pkl"
        
    def is_audio_file(self, filename):
        """Check if file has supported audio extension"""
        return filename.lower().endswith(self.supported_formats)
    
    def load_audio_file(self, file_path):
        """Load audio file with robust MP3 support and error handling"""
        try:
            # Try librosa first (works for WAV and some MP3 files)
            if file_path.lower().endswith('.wav') or self.use_librosa_for_mp3:
                try:
                    duration = librosa.get_duration(filename=file_path)
                    duration = min(duration, self.max_duration)
                    
                    audio, sr = librosa.load(
                        file_path,
                        sr=self.target_sr,
                        duration=duration,
                        mono=True,
                        res_type='kaiser_fast'
                    )
                    return audio, True
                except Exception as librosa_error:
                    if file_path.lower().endswith('.mp3'):
                        print(f"Librosa failed for MP3, trying soundfile: {file_path}")
                        self.use_librosa_for_mp3 = False  # Don't try librosa for next MP3s
                        return self.load_audio_file(file_path)  # Retry with soundfile
                    raise librosa_error
            
            # Fallback to soundfile for problematic MP3s
            try:
                # Get duration using soundfile
                with sf.SoundFile(file_path) as f:
                    duration = min(len(f) / f.samplerate, self.max_duration)
                    target_frames = int(duration * self.target_sr)
                    
                    # Read the audio
                    audio = f.read(frames=target_frames, dtype='float32', always_2d=True)
                    
                    # Convert to mono if needed
                    if audio.ndim > 1 and audio.shape[1] > 1:
                        audio = np.mean(audio, axis=1)
                    else:
                        audio = audio.flatten()
                    
                    # Resample if needed
                    if f.samplerate != self.target_sr:
                        audio = librosa.resample(audio, orig_sr=f.samplerate, target_sr=self.target_sr)
                    
                    # Pad if needed
                    if len(audio) < self.target_sr * self.max_duration:
                        padding = self.target_sr * self.max_duration - len(audio)
                        audio = np.pad(audio, (0, padding), mode='constant')
                    
                    return audio, True
            except Exception as sf_error:
                print(f"Soundfile failed for {file_path}: {str(sf_error)}")
                return None, False
                
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None, False
    
    def load_dataset(self, root_path, use_cache=True):
        """Load audio files with better error handling and progress tracking"""
        cache_file = os.path.join(self.cache_dir, self._get_cache_key(root_path))
        
        # Try to load from cache
        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    return cached_data['features'], cached_data['labels'], cached_data['file_paths']
            except Exception as e:
                print(f"Error loading cache: {str(e)}. Re-processing data.")
        
        # Process data if cache not available
        features = []
        labels = []
        file_paths = []
        error_count = 0
        total_files = 0
        
        # First count total files
        for label in os.listdir(root_path):
            label_path = os.path.join(root_path, label)
            if os.path.isdir(label_path):
                total_files += len([f for f in os.listdir(label_path) if self.is_audio_file(f)])
        
        processed_files = 0
        start_time = time.time()
        
        for label in os.listdir(root_path):
            label_path = os.path.join(root_path, label)
            if not os.path.isdir(label_path):
                continue
                
            print(f"\nProcessing category: {label}...")
            
            for file in os.listdir(label_path):
                if not self.is_audio_file(file):
                    continue
                    
                file_path = os.path.join(label_path, file)
                processed_files += 1
                
                # Show progress
                if processed_files % 10 == 0:
                    elapsed = time.time() - start_time
                    remaining = (elapsed/processed_files) * (total_files-processed_files)
                    print(f"  Progress: {processed_files}/{total_files} "
                          f"({remaining:.1f}s remaining)")
                
                audio, success = self.load_audio_file(file_path)
                
                if not success:
                    error_count += 1
                    continue
                    
                try:
                    extracted_features = self.extract_features(audio)
                    if extracted_features is not None:
                        features.append(extracted_features)
                        labels.append(label)
                        file_paths.append(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    error_count += 1
        
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        # Save to cache
        if use_cache and len(features) > 0:  # Only cache if we have data
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'features': features,
                        'labels': labels,
                        'file_paths': file_paths
                    }, f)
                print(f"Saved processed data to cache: {cache_file}")
            except Exception as e:
                print(f"Error saving cache: {str(e)}")
        
        print(f"\nCompleted loading. Total errors: {error_count}/{total_files}")
        return features, labels, file_paths

    def extract_features(self, audio):
        """Extract comprehensive audio features with robustness checks"""
        try:
            features = []
            
            # MFCCs with delta features
            mfccs = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=self.n_mfcc)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            features.extend(np.mean(delta_mfccs, axis=1))
            features.extend(np.mean(delta2_mfccs, axis=1))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.target_sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.target_sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.target_sr)
            
            features.extend([
                np.mean(spectral_centroid), np.std(spectral_centroid),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                *np.mean(spectral_contrast, axis=1),
                np.mean(spectral_rolloff), np.std(spectral_rolloff)
            ])
            
            # Temporal features
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            rmse = librosa.feature.rms(y=audio)
            tempogram = librosa.feature.tempogram(y=audio, sr=self.target_sr)
            
            features.extend([
                np.mean(zero_crossing_rate), np.std(zero_crossing_rate),
                np.mean(rmse), np.std(rmse),
                np.mean(tempogram), np.std(tempogram)
            ])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.target_sr)
            features.extend(np.mean(chroma, axis=1))
            features.extend(np.std(chroma, axis=1))
            
            return features
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None
    
    def preprocess_data(self, X, y):
        """Encode labels and scale features with validation"""
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No valid data to preprocess")
            
        # Encode labels
        y_encoded = self.le.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y_categorical, self.le.classes_

# --------------------------
# 2. Enhanced Model Architecture
# --------------------------

class BioGeoClassifier:
    """Advanced classifier with improved architecture"""
    
    def __init__(self, input_shape, num_classes):
        self.model = self.build_enhanced_model(input_shape, num_classes)
        
    def build_enhanced_model(self, input_shape, num_classes):
        """Construct improved neural network with residual connections"""
        inputs = layers.Input(shape=input_shape)
        
        # Initial dense layer
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Residual block 1
        residual = x
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.add([x, residual])
        
        # Attention mechanism
        attention = layers.Dense(256, activation='tanh')(x)
        attention = layers.Dense(1, activation='softmax')(attention)
        attention = layers.Flatten()(attention)
        attention = layers.RepeatVector(512)(attention)
        attention = layers.Permute([2, 1])(attention)
        x = layers.Multiply()([x, attention])
        
        # Residual block 2
        residual = x
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.add([x, residual])
        
        # Output
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Custom optimizer configuration
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Enhanced training with callbacks"""
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history

# --------------------------
# 3. Enhanced Evaluation & Visualization
# --------------------------

class ResultVisualizer:
    """Enhanced visualization with more metrics"""
    
    @staticmethod
    def plot_training_history(history):
        """Plot training and validation metrics"""
        metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
        plt.figure(figsize=(15, 12))
        
        for i, metric in enumerate(metrics):
            plt.subplot(3, 2, i+1)
            plt.plot(history.history[metric], label=f'Train {metric}')
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'Model {metric.capitalize()}')
            plt.ylabel(metric.capitalize())
            plt.xlabel('Epoch')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names):
        """Enhanced confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.show()
    
    @staticmethod
    def plot_spectral_features(audio_path, sr=22050):
        """Enhanced spectral feature visualization"""
        y, sr = librosa.load(audio_path, sr=sr)
        
        plt.figure(figsize=(15, 10))
        
        # Waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title('Audio Waveform')
        
        # Spectrogram
        plt.subplot(3, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log-frequency power spectrogram')
        
        # MFCCs
        plt.subplot(3, 1, 3)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        
        plt.tight_layout()
        plt.show()

# --------------------------
# 4. Enhanced Spectrum Management System
# --------------------------

class SpectrumManager:
    """Enhanced spectrum allocation with priority management"""
    
    def __init__(self, classifier, label_encoder):
        self.classifier = classifier
        self.label_encoder = label_encoder
        self.band_allocation = {
            'infrasonic': (1, 20),    # Earthquakes
            'low': (20, 500),        # Hurricanes, waves
            'mid': (500, 20000),      # Marine mammals
            'high': (20000, 40000)    # Backup
        }
        self.current_allocations = []
        self.priority_map = {
            'dolphin': 3,    # Highest priority
            'whale': 3,
            'minke_whale': 3,
            'hurricane': 2,
            'earthquake': 1,  # Lowest priority
            'wave': 1
        }
        
    def analyze_audio(self, audio):
        """Enhanced audio analysis with error handling"""
        try:
            # Extract features
            loader = AudioDataLoader()
            features = loader.extract_features(audio)
            if features is None:
                return None, None
                
            features = np.array(features).reshape(1, -1)
            
            # Classify
            pred = self.classifier.predict(features)
            class_idx = np.argmax(pred)
            sound_class = self.label_encoder[class_idx]
            
            # Frequency analysis
            freqs = np.fft.rfftfreq(len(audio), d=1/22050)
            fft = np.abs(np.fft.rfft(audio))
            
            band_energy = {}
            for band, (low, high) in self.band_allocation.items():
                mask = (freqs >= low) & (freqs <= high)
                band_energy[band] = np.sum(fft[mask])
            
            return sound_class, band_energy
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return None, None
    
    def allocate_band(self, sound_class, band_energy):
        """Enhanced allocation with priority system"""
        if sound_class not in self.priority_map:
            sound_class = 'other'
            
        allocation = {
            'primary': None,
            'secondary': None,
            'priority': self.priority_map.get(sound_class, 0)
        }
        
        # Priority-based allocation
        if sound_class in ['dolphin', 'whale', 'minke_whale']:
            allocation['primary'] = 'mid'
            allocation['secondary'] = 'high'
        elif sound_class == 'earthquake':
            allocation['primary'] = 'infrasonic'
        elif sound_class == 'hurricane':
            # Dynamic allocation based on current usage
            if band_energy['mid'] > 0.5 * band_energy['low']:
                allocation['primary'] = 'low'
            else:
                allocation['primary'] = 'mid'
        else:
            # Default allocation for unknown classes
            allocation['primary'] = 'low'
            
        return allocation
    
    def visualize_allocation(self):
        """Enhanced visualization with priority information"""
        if not self.current_allocations:
            print("No allocations to visualize")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Create a color map based on priority
        colors = {1: 'lightblue', 2: 'orange', 3: 'red'}
        
        for alloc in self.current_allocations:
            band_range = self.band_allocation[alloc['band']]
            plt.barh(
                alloc['class'], 
                width=band_range[1]-band_range[0],
                left=band_range[0],
                alpha=0.6,
                color=colors.get(alloc['priority'], 'gray'),
                label=f"{alloc['class']} (Priority {alloc['priority']})"
            )
        
        plt.title('Current Spectrum Allocation with Priority')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Sound Class')
        plt.xscale('log')
        plt.grid(True, which="both", ls="--")
        
        # Create custom legend
        handles = [
            plt.Rectangle((0,0),1,1, color='red', alpha=0.6, label='High Priority (Marine Mammals)'),
            plt.Rectangle((0,0),1,1, color='orange', alpha=0.6, label='Medium Priority (Weather)'),
            plt.Rectangle((0,0),1,1, color='lightblue', alpha=0.6, label='Low Priority (Geophonic)')
        ]
        plt.legend(handles=handles)
        
        plt.show()

# --------------------------
# 5. Main Execution with Enhanced Features
# --------------------------

def main():
    print("=== Marine Bioacoustic and Geophonic Sound Classification ===")
    
    # 1. Initialize data loader with enhanced parameters
    loader = AudioDataLoader(
        target_sr=44100,  # Higher sampling rate for better frequency resolution
        max_duration=4,   # Slightly shorter for more consistent samples
        n_mfcc=60        # More MFCC coefficients for better feature representation
    )
    
    # 2. Load datasets with progress feedback
    print("\nLoading biophonic dataset...")
    bio_X, bio_y, bio_paths = loader.load_dataset("/Users/fatbatman/Develop/Liquid Neural Network/dataset")
    
    print("\nLoading geophonic dataset...")
    geo_X, geo_y, geo_paths = loader.load_dataset("/Users/fatbatman/Downloads/Dataset")
    
    if len(bio_X) == 0 or len(geo_X) == 0:
        print("Error: No valid data loaded. Check your dataset paths and file formats.")
        return
    
    # 3. Combine and preprocess data
    print("\nPreprocessing data...")
    X = np.concatenate((bio_X, geo_X))
    y = np.concatenate((bio_y, geo_y))
    
    try:
        X_scaled, y_categorical, class_names = loader.preprocess_data(X, y)
    except ValueError as e:
        print(f"Preprocessing error: {str(e)}")
        return
    
    # 4. Split data with stratification
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, 
        test_size=0.2, 
        random_state=42,
        stratify=y_categorical
    )
    
    # 5. Train model with enhanced architecture
    print("\nTraining model...")
    classifier = BioGeoClassifier(
        input_shape=(X_train.shape[1],), 
        num_classes=len(class_names)
    )
    
    history = classifier.train(
        X_train, y_train,
        X_test, y_test,
        epochs=150,  # Increased epochs for better convergence
        batch_size=64  # Larger batch size for stability
    )
    
    # 6. Comprehensive evaluation
    print("\nEvaluating model performance...")
    y_pred = classifier.model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    
    print(f"\n=== Final Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # 7. Enhanced visualizations
    visualizer = ResultVisualizer()
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)
    
    # 8. Spectrum management demonstration
    print("\n=== Spectrum Management Demonstration ===")
    spectrum_manager = SpectrumManager(classifier.model, class_names)
    
    # Analyze sample files
    sample_paths = [
        np.random.choice(bio_paths),  # Random biophonic sample
        np.random.choice(geo_paths)   # Random geophonic sample
    ]
    
    for path in sample_paths:
        print(f"\nAnalyzing: {os.path.basename(path)}")
        
        # Load and analyze
        audio, _ = librosa.load(path, sr=44100, duration=4)
        sound_class, band_energy = spectrum_manager.analyze_audio(audio)
        
        if sound_class is None:
            print("Analysis failed for this sample")
            continue
            
        allocation = spectrum_manager.allocate_band(sound_class, band_energy)
        
        print(f"Classified as: {sound_class}")
        print("Band Energy Distribution:")
        for band, energy in band_energy.items():
            print(f"  {band}: {energy:.2f}")
        print("Allocation:", allocation)
        
        # Add to current allocations
        spectrum_manager.current_allocations.append({
            'class': sound_class,
            'band': allocation['primary'],
            'priority': allocation['priority'],
            'energy': band_energy[allocation['primary']]
        })
        
        # Visualize sample features
        visualizer.plot_spectral_features(path)
    
    # Show final spectrum allocation
    spectrum_manager.visualize_allocation()

if __name__ == "__main__":
    import tensorflow as tf
    main()