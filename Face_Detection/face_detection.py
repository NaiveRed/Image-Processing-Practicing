import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

cv_path = os.path.dirname(cv.__file__)
data_path = os.path.join(os.getcwd(), "data")
face_cascade = cv.CascadeClassifier(os.path.join(
    cv_path, "data", 'haarcascade_frontalface_default.xml'))
eye_cascade = cv.CascadeClassifier(os.path.join(
    cv_path, "data", 'haarcascade_eye_tree_eyeglasses.xml'))
smile_cascade = cv.CascadeClassifier(os.path.join(
    cv_path, "data", 'haarcascade_smile.xml'))

img = cv.imread(os.path.join(data_path, "img.jpg"))


def main():

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    vis = np.copy(img)

    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    # detect face
    for (x, y, w, h) in faces:
        # region of interest
        cv.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]  # 縮小區塊來偵測其他部位
        roi_color = vis[y:y+h, x:x+w]  # 用來在原圖標記區域
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        smiles = smile_cascade.detectMultiScale(
            roi_gray, 1.1, 3, minSize=(100, 50), maxSize=(200, 100))

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        for (sx, sy, sw, sh) in smiles:
            flag = False
            for (ex, ey, ew, eh) in eyes:
                if sx < ex+ew//2 < sx+sw and sy < ey+eh//2 < sy+sh:
                    flag = True
                    break
            if flag:
                continue
            cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

    vis = cv.cvtColor(vis, cv.COLOR_BGR2RGB)
    plt.figure(num="Test",)
    # plt.xticks([]), plt.yticks([])
    plt.title("Pic")
    plt.tight_layout()
    plt.imshow(vis, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
