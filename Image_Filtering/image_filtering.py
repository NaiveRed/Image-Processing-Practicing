# coding=utf-8
import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

ONLY_FACE = False  # True: filter only affect part of face. False: affect whole image
SHOW_ROI = True  # only available in (ONLY_FACE=True)

# source img
data_path = os.path.join(os.getcwd(), "data")
imgs = [cv.imread(os.path.join(data_path, "img1.jpg")),
        cv.imread(os.path.join(data_path, "img2.jpg")),
        cv.imread(os.path.join(data_path, "img3.jpg"))]

titles = ["Origin", "3x3 Sharpen", "Gaussian blur", "Bilateral filter"]

# face detection
cv_path = os.path.dirname(cv.__file__)
face_cascade = cv.CascadeClassifier(os.path.join(
    cv_path, "data", 'haarcascade_frontalface_default.xml'))


def face_detect(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    return faces


def set_image(I, p, name, fig_n):
    plt.figure(num=fig_n)
    plt.subplot(p)
    plt.imshow(cv.cvtColor(I, cv.COLOR_BGR2RGB))
    plt.title(name)
    plt.xticks([])
    plt.yticks([])


def main():

    # 3x3 Sharpen filter
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], np.float32)
    # res = cv.filter2D(img, ddepth=cv.CV_8U, kernel=sharpen_kernel)

    # Gaussian blur 3x3
    blur_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], np.float32)/16
    # res = cv.filter2D(img, ddepth=cv.CV_8U, kernel=blur_kernel)

    # Bilateral filter(雙邊濾波)
    # res = cv.bilateralFilter(img, 5, 75, 75)

    for n, I in zip(range(len(imgs)), imgs):

        s = "Face pic"+str(n+1)
        plt.figure(num=s, tight_layout=True)

        if ONLY_FACE:
            res = [I.copy() for _ in range(4)]
            faces = face_detect(I)

            for (x, y, w, h) in faces:

                if ONLY_FACE:
                    # Sharpen filter
                    res[1][y:y+h, x:x+w] = cv.filter2D(
                        res[1][y:y+h, x:x+w], ddepth=cv.CV_8U, kernel=sharpen_kernel)
                    # Gaussian blur
                    res[2][y:y+h, x:x+w] = cv.filter2D(
                        res[2][y:y+h, x:x+w], ddepth=cv.CV_8U, kernel=blur_kernel)
                    # Bilateral filte
                    res[3][y:y+h, x:x + w] = cv.bilateralFilter(
                        res[3][y:y+h, x:x+w], 5, 75, 75)

            for (x, y, w, h) in faces:

                if SHOW_ROI:
                    cv.rectangle(res[1], (x, y), (x+w, y+h),
                                 (255, 0, 0), thickness=2)
                    cv.rectangle(res[2], (x, y), (x+w, y+h),
                                 (255, 0, 0), thickness=2)
                    cv.rectangle(res[3], (x, y), (x+w, y+h),
                                 (255, 0, 0), thickness=2)

        else:
            res = [I,
                   cv.filter2D(I, ddepth=cv.CV_8U, kernel=sharpen_kernel),
                   cv.filter2D(I, ddepth=cv.CV_8U, kernel=blur_kernel),
                   cv.bilateralFilter(I, 5, 75, 75)]

        for i in range(4):
            set_image(res[i], "22"+str(i+1), titles[i], s)

    plt.show()


if __name__ == "__main__":
    main()
