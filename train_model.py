import numpy as np
import os
import pickle
from satellite_features import extract_features, CLASSES

def build_training_data(train_path):
    X_train = []
    y_train = []
    
    print("Building training dataset...")
    for class_idx, class_name in enumerate(CLASSES):
        folder = os.path.join(train_path, class_name)
        if not os.path.isdir(folder):
            continue
        
        count = 0
        for file in os.listdir(folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                img_path = os.path.join(folder, file)
                features = extract_features(img_path)
                
                if features is not None:
                    X_train.append(features)
                    y_train.append(class_idx)
                    count += 1
        
        print(f"  {class_name}: {count} samples")
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-7
    X_train = (X_train - mean) / std
    
    return X_train, y_train, mean, std

def Euclidean_distance(train_path, E=9):
    X_train, y_train, mean, std = build_training_data(train_path)
    
    print(f"\nPreparing classifier (E={E})...")
    print(f"Total training samples: {len(X_train)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    model_data = {
        'X_train': X_train,
        'y_train': y_train,
        'mean': mean,
        'std': std,
        'E': E
    }
    
    with open('Euclidean_distance.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved successfully!")
    return X_train, y_train, mean, std, E
if __name__ == "__main__":
    TRAIN_PATH = r"F:\assignment\DIP\Lecture\EuroSAT Dataset\Train"
    Euclidean_distance(TRAIN_PATH, E=9)