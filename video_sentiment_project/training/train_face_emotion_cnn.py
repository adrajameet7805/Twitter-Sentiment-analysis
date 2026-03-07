import os
import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Emotion labels mapping
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

def load_dataset(csv_path):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    X = []
    y = []
    usages = []
    
    for _, row in df.iterrows():
        pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.float32)
        pixels = pixels.reshape((48, 48, 1))
        pixels = pixels / 255.0  # Normalize
        
        X.append(pixels)
        y.append(row['emotion'])
        
        if 'Usage' in row:
            usages.append(row['Usage'])
            
    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=7)
    
    if usages:
        usages = np.array(usages)
        train_idx = usages == 'Training'
        val_idx = usages == 'PublicTest'
        test_idx = usages == 'PrivateTest'
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
    else:
        # Fallback if Usage column is missing
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_cnn_model():
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(7))
    model.add(Activation('softmax'))

    return model

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'fer2013.csv')
    model_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    os.makedirs(model_dir, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(data_path)

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    print("Building CNN model...")
    model = build_cnn_model()
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    model.summary()

    model_path = os.path.join(model_dir, 'face_emotion_cnn.h5')
    labels_path = os.path.join(model_dir, 'emotion_labels.json')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    print("Evaluating model...")
    # model.fit saves the best model during checkpointing, but we also restored best weights from EarlyStopping
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    conf_matrix = confusion_matrix(y_true, y_pred)
    
    target_names = [EMOTION_LABELS[i] for i in range(7)]
    class_report = classification_report(y_true, y_pred, target_names=target_names)

    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Save emotional labels
    with open(labels_path, 'w') as f:
        json.dump(EMOTION_LABELS, f)

    print(f"Model saved to {model_path}")
    print(f"Labels saved to {labels_path}")

if __name__ == '__main__':
    main()
