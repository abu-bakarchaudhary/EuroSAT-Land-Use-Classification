import matplotlib.pyplot as plt
import numpy as np
import pickle
from satellite_features import CLASSES

def plot_visualizations():
    print("Loading training data for visualization...")
    with open('Euclidean_distance.pkl', 'rb') as f:
        model_data = pickle.load(f)
        
    X_train = model_data['X_train']
    y_train = model_data['y_train']
    feature_hue = 0        
    feature_edge = 17      

    print("Generating Scatter Plot...")
    plt.figure(figsize=(12, 8))
    
    # 1. SCATTER PLOT (Mean Hue vs Edge Density)
    for class_idx, class_name in enumerate(CLASSES):
        class_mask = (y_train == class_idx)
        plt.scatter(X_train[class_mask, feature_hue], 
                    X_train[class_mask, feature_edge], 
                    label=class_name, alpha=0.7, s=35, edgecolors='none')
        
    plt.title('Feature Separability: Mean Hue vs Edge Density', fontsize=16)
    plt.xlabel('Normalized Mean Hue', fontsize=12)
    plt.ylabel('Normalized Edge Density', fontsize=12)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    print("Generating Box Plot...")
    # 2. BOX PLOT (Distribution of Edge Density across all 10 classes)
    plt.figure(figsize=(14, 8))
    
    data_to_plot = []
    for class_idx in range(len(CLASSES)):
        class_mask = (y_train == class_idx)
        data_to_plot.append(X_train[class_mask, feature_edge])
        
    bplot = plt.boxplot(data_to_plot, patch_artist=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    plt.xticks(ticks=range(1, len(CLASSES) + 1), labels=CLASSES, rotation=45, ha='right', fontsize=11)
    plt.title('Edge Density Distribution Across Land Cover Classes', fontsize=16)
    plt.xlabel('Land Cover Class', fontsize=12)
    plt.ylabel('Normalized Edge Density', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()

    print("-> Displaying plots on screen... (Close the windows to end the script)")
    
    plt.show()
if __name__ == "__main__":
    plot_visualizations()