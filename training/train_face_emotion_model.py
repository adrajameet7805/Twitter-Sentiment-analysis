import os
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog

def load_data(csv_path):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    X = []
    y = []
    usages = []
    
    for _, row in df.iterrows():
        pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.float32)
        pixels = pixels.reshape((48, 48))
        pixels = pixels / 255.0
        
        X.append(pixels)
        y.append(row['emotion'])
        
        if 'Usage' in row:
            usages.append(row['Usage'])
            
    return np.array(X), np.array(y), np.array(usages) if usages else None

def extract_features(images, hog_config):
    print("Extracting HOG features...")
    features = []
    for img in images:
        feat = hog(
            img,
            orientations=hog_config['orientations'],
            pixels_per_cell=hog_config['pixels_per_cell'],
            cells_per_block=hog_config['cells_per_block'],
            block_norm='L2-Hys',
            visualize=False,
            channel_axis=None
        )
        features.append(feat)
    return np.array(features)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'fer2013.csv')
    model_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    os.makedirs(model_dir, exist_ok=True)

    X, y, usages = load_data(data_path)
    
    if usages is not None:
        train_idx = usages == 'Training'
        val_idx = usages == 'PublicTest'
        test_idx = usages == 'PrivateTest'
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    hog_config = {
        'orientations': 9,
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2)
    }

    X_train_features = extract_features(X_train, hog_config)
    X_val_features = extract_features(X_val, hog_config)
    X_test_features = extract_features(X_test, hog_config)

    print("Training SVM classifier...")
    clf = SVC(kernel='linear', probability=True, random_state=42)
    clf.fit(X_train_features, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_test_features)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    target_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    class_report = classification_report(y_test, y_pred, target_names=target_names)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    svm_model_path = os.path.join(model_dir, 'face_emotion_svm.pkl')
    hog_config_path = os.path.join(model_dir, 'hog_feature_config.pkl')

    print("Saving models...")
    with open(svm_model_path, 'wb') as f:
        pickle.dump(clf, f)

    with open(hog_config_path, 'wb') as f:
        pickle.dump(hog_config, f)

    print(f"Models saved successfully to {model_dir}")

if __name__ == '__main__':
    main()
