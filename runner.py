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
warnings.filterwarnings('ignore')

# --------------------------
# 1. Data Loading & Preprocessing
# --------------------------

class AudioDataLoader:
    """Handles loading and preprocessing of audio datasets"""
    
    def __init__(self, target_sr=22050, duration=5, n_mfcc=40):
        self.target_sr = target_sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.le = LabelEncoder()
        
    def load_dataset(self, root_path):
        """Load audio files from nested folder structure"""
        features = []
        labels = []
        file_paths = []
        
        for label in os.listdir(root_path):
            label_path = os.path.join(root_path, label)
            if not os.path.isdir(label_path):
                continue
                
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                try:
                    audio, sr = librosa.load(file_path, 
                                           sr=self.target_sr, 
                                           duration=self.duration)
                    features.append(self.extract_features(audio))
                    labels.append(label)
                    file_paths.append(file_path)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return np.array(features), np.array(labels), file_paths
    
    def extract_features(self, audio):
        """Extract comprehensive audio features"""
        features = []
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=self.n_mfcc)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.target_sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.target_sr)
        features.extend([
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            *np.mean(spectral_contrast, axis=1)
        ])
        
        # Temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        rmse = librosa.feature.rms(y=audio)
        features.extend([
            np.mean(zero_crossing_rate), np.std(zero_crossing_rate),
            np.mean(rmse), np.std(rmse)
        ])
        
        return features
    
    def preprocess_data(self, X, y):
        """Encode labels and scale features"""
        # Encode labels
        y_encoded = self.le.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y_categorical, self.le.classes_

# --------------------------
# 2. Model Architecture
# --------------------------

class BioGeoClassifier:
    """Advanced classifier for biophonic and geophonic sounds"""
    
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)
        
    def build_model(self, input_shape, num_classes):
        """Construct neural network with attention mechanism"""
        inputs = layers.Input(shape=input_shape)
        
        # Feature processing
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Attention layer
        attention = layers.Dense(256, activation='tanh')(x)
        attention = layers.Dense(1, activation='softmax')(attention)
        attention = layers.Flatten()(attention)
        attention = layers.RepeatVector(512)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        x = layers.Multiply()([x, attention])
        
        # Output
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train model with early stopping"""
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history

# --------------------------
# 3. Evaluation & Visualization
# --------------------------

class ResultVisualizer:
    """Handles performance visualization and analysis"""
    
    @staticmethod
    def plot_training_history(history):
        """Plot training and validation metrics"""
        plt.figure(figsize=(15, 5))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names):
        """Plot detailed confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    @staticmethod
    def plot_spectrogram(audio_path, sr=22050):
        """Visualize spectrogram of a sample audio"""
        y, sr = librosa.load(audio_path, sr=sr)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max),
                                sr=sr, y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.show()

# --------------------------
# 4. Spectrum Management System
# --------------------------

class SpectrumManager:
    """Handles spectrum allocation and sharing"""
    
    def __init__(self, classifier, label_encoder):
        self.classifier = classifier
        self.label_encoder = label_encoder
        self.band_allocation = {
            'infrasonic': (1, 20),    # Earthquakes
            'low': (20, 500),         # Hurricanes
            'mid': (500, 20000),      # Marine mammals
            'high': (20000, 40000)    # Backup
        }
        self.current_allocations = []
        
    def analyze_audio(self, audio):
        """Classify audio and analyze frequency content"""
        # Extract features
        loader = AudioDataLoader()
        features = loader.extract_features(audio)
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
    
    def allocate_band(self, sound_class, band_energy):
        """Determine optimal frequency band allocation"""
        allocation = {}
        
        # Priority rules
        if sound_class in ['dolphin', 'whale']:
            allocation['primary'] = 'mid'
            allocation['secondary'] = 'high'
        elif sound_class == 'earthquake':
            allocation['primary'] = 'infrasonic'
        elif sound_class == 'hurricane':
            # Avoid mid-band if already occupied
            if band_energy['mid'] > 0.3 * band_energy['low']:
                allocation['primary'] = 'low'
            else:
                allocation['primary'] = 'mid'
        
        return allocation
    
    def visualize_allocation(self):
        """Plot current spectrum allocations"""
        plt.figure(figsize=(10, 6))
        for alloc in self.current_allocations:
            band_range = self.band_allocation[alloc['band']]
            plt.barh(alloc['class'], 
                     width=band_range[1]-band_range[0],
                     left=band_range[0],
                     alpha=0.6,
                     label=f"{alloc['class']} ({alloc['band']})")
        
        plt.title('Current Spectrum Allocation')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Sound Class')
        plt.xscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()

# --------------------------
# 5. Main Execution
# --------------------------

def main():
    # 1. Load and preprocess data
    print("Loading datasets...")
    loader = AudioDataLoader()
    
    # Load biophonic data
    bio_X, bio_y, bio_paths = loader.load_dataset("'/Users/fatbatman/Develop/Liquid Neural Network/dataset'")
    
    # Load geophonic data
    geo_X, geo_y, geo_paths = loader.load_dataset("/Users/fatbatman/Downloads/Dataset")
    
    # Combine datasets
    X = np.concatenate((bio_X, geo_X))
    y = np.concatenate((bio_y, geo_y))
    
    # Preprocess
    X_scaled, y_categorical, class_names = loader.preprocess_data(X, y)
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, test_size=0.2, random_state=42
    )
    
    # 3. Train model
    print("\nTraining model...")
    classifier = BioGeoClassifier(input_shape=(X_train.shape[1],), 
                                num_classes=len(class_names))
    history = classifier.train(X_train, y_train, 
                             X_test, y_test, 
                             epochs=100)
    
    # 4. Evaluate
    print("\nEvaluating model...")
    y_pred = classifier.model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # 5. Visualizations
    visualizer = ResultVisualizer()
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)
    
    # 6. Spectrum Management Demo
    print("\nDemonstrating spectrum management...")
    spectrum_manager = SpectrumManager(classifier.model, class_names)
    
    # Sample analysis
    sample_paths = [
        np.random.choice(bio_paths),  # Random biophonic sample
        np.random.choice(geo_paths)   # Random geophonic sample
    ]
    
    for path in sample_paths:
        audio, _ = librosa.load(path, sr=22050, duration=5)
        sound_class, band_energy = spectrum_manager.analyze_audio(audio)
        allocation = spectrum_manager.allocate_band(sound_class, band_energy)
        
        print(f"\nSample: {os.path.basename(path)}")
        print(f"Classified as: {sound_class}")
        print("Band Energy Distribution:")
        for band, energy in band_energy.items():
            print(f"  {band}: {energy:.2f}")
        print("Allocation:", allocation)
        
        spectrum_manager.current_allocations.append({
            'class': sound_class,
            'band': allocation['primary'],
            'energy': band_energy[allocation['primary']]
        })
        
        visualizer.plot_spectrogram(path)
    
    spectrum_manager.visualize_allocation()

if __name__ == "__main__":
    main()