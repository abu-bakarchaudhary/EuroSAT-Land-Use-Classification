import cv2
import numpy as np

CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]
def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None: 
        return None
    
    # Preprocessing
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    
    features = []
    
    # 1. Color Features (HSV statistics)
    for i in range(3): 
        channel = hsv[:, :, i]
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.percentile(channel, 25),
            np.percentile(channel, 75)
        ])
    
    # 2. Texture Features
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(np.var(laplacian))
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    features.extend([
        np.mean(np.abs(sobelx)),
        np.mean(np.abs(sobely)),
        np.std(sobelx),
        np.std(sobely)
    ])
    
    # 3. Edge Features
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)
    
    gradient_direction = np.arctan2(sobely, sobelx)
    hist, _ = np.histogram(gradient_direction, bins=4, range=(-np.pi, np.pi))
    hist = hist / (hist.sum() + 1e-7)
    features.extend(hist)
    
    # 4. Intensity Features
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.percentile(gray, 10),
        np.percentile(gray, 90)
    ])
    
    # 5. Local Binary Pattern (simplified)
    lbp_sum = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(gray, dx, axis=0), dy, axis=1)
            lbp_sum += (shifted > gray).astype(float)
    features.append(np.mean(lbp_sum))
    
    return np.array(features, dtype=np.float32)