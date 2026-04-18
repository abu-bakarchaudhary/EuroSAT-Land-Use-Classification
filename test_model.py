import numpy as np
import os
import pickle
from satellite_features import extract_features, CLASSES
from visualization import plot_visualizations
def load_model():
    with open('Euclidean_distance.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    return (model_data['X_train'], model_data['y_train'], 
            model_data['mean'], model_data['std'], model_data['E'])

def Euclidean_distance(X_train, y_train, features, mean, std, E):
    features_norm = (features - mean) / std
    
    # Calculate Euclidean distances to all training samples
    distances = np.sqrt(np.sum((X_train - features_norm)**2, axis=1))
    
    E_nearest_indices = np.argsort(distances)[:E]
    E_nearest_labels = y_train[E_nearest_indices]
    
    unique, counts = np.unique(E_nearest_labels, return_counts=True)
    predicted_class = unique[np.argmax(counts)]
    
    return predicted_class

def evaluate_model(test_path):
    X_train, y_train, mean, std, E = load_model()
    
    confusion_matrix = np.zeros((10, 10), dtype=int)
    correct_per_class = {c: 0 for c in CLASSES}
    total_per_class = {c: 0 for c in CLASSES}
    total_correct = 0
    
    print(f"Evaluating manual Euclidean_distance(E={E}) on test set...\n")
    
    for true_idx, true_label in enumerate(CLASSES):
        folder = os.path.join(test_path, true_label)
        if not os.path.isdir(folder):
            continue
        
        for file in os.listdir(folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                img_path = os.path.join(folder, file)
                features = extract_features(img_path)
                
                if features is None:
                    continue
                
                pred_idx = Euclidean_distance(X_train, y_train, features, mean, std, E)
                
                confusion_matrix[true_idx, pred_idx] += 1
                total_per_class[true_label] += 1
                
                if true_idx == pred_idx:
                    correct_per_class[true_label] += 1
                    total_correct += 1
    
    total_images = sum(total_per_class.values())
    overall_acc = (total_correct / total_images) * 100 if total_images > 0 else 0
    
    print("="*70)
    print("PER-CLASS ACCURACY")
    print("="*70)
    print(f"{'Class Name':<25} | {'Accuracy (%)':<15} | {'Correct/Total'}")
    print("-" * 70)
    
    class_accuracies = []
    for c in CLASSES:
        total = total_per_class[c]
        if total > 0:
            acc = (correct_per_class[c] / total) * 100
            class_accuracies.append(acc)
            print(f"{c:<25} | {acc:>14.2f}% | {correct_per_class[c]:>4}/{total:<4}")
        else:
            print(f"{c:<25} | {'N/A':>14} | 0/0")
    
    print("-" * 70)
    print(f"{'OVERALL ACCURACY':<25} | {overall_acc:>14.2f}% | {total_correct:>4}/{total_images:<4}")
    print(f"{'AVERAGE CLASS ACCURACY':<25} | {np.mean(class_accuracies):>14.2f}% |")
    
    print("\n" + "="*85)
    print("CONFUSION MATRIX")
    print("="*85)
    
    header = f"{'True \\ Pred':<15}" + "".join([f"{c[:4]:>6}" for c in CLASSES])
    print(header)
    print("-" * 85)
    
    for i, true_label in enumerate(CLASSES):
        row_str = f"{true_label[:12]:<15}"
        for j in range(10):
            row_str += f"{confusion_matrix[i, j]:>6}"
        print(row_str)
    
    print("\n")
if __name__ == "__main__":
    TEST_PATH = r"F:\assignment\DIP\Lecture\EuroSAT Dataset\Test"
    evaluate_model(TEST_PATH)
    plot_visualizations() 