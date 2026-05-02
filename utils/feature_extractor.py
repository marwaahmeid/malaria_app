import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

def extract_features(img):
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG
    hog_feat = hog(gray, pixels_per_cell=(8,8), cells_per_block=(2,2))

    # LBP
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0,11))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)

    # Color histogram
    color_feat = []
    for c in cv2.split(img):
        hist = cv2.calcHist([c],[0],None,[32],[0,256])
        color_feat.extend(cv2.normalize(hist,hist).flatten())

    feat = np.hstack([hog_feat, lbp_hist, color_feat])

    return feat