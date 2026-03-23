import cv2
import glob
import numpy as np
import os

def image_colorfulness(img):
    # img is BGR from cv2.imread
    (B, G, R) = cv2.split(img.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)

    rbMean, rbStd = (np.mean(rg), np.std(rg))
    ybMean, ybStd = (np.mean(yb), np.std(yb))

    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    return stdRoot + (0.3 * meanRoot)

def evaluate_folder(folder):
    paths = glob.glob(os.path.join(folder, '*.jpg'))
    scores = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        scores.append(image_colorfulness(img))
    return np.mean(scores) if scores else 0.0

if __name__ == "__main__":
    gen_folder = os.path.join("evaluation_dataset", "generated")
    avg_c = evaluate_folder(gen_folder)
    print(f"Average colorfulness (generated): {avg_c:.3f}")