# coding=utf-8
'''
部分算法參考：
https://github.com/machinelearninggod/Image-Processing-Algorithms/blob/master/histogram_match.py
'''
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

data_path = os.path.join(os.getcwd(), "data")
name = "img1"
In = cv2.imread(os.path.join(data_path, name+".jpg"), cv2.IMREAD_GRAYSCALE)

plt.figure(num="Power-Law", tight_layout=True)
plt.figure(num="Histogram Equalization", tight_layout=True)
plt.figure(num="Histogram Matching", tight_layout=True)
fig_num = plt.get_fignums()


def set_image(I, p, name, fig_n):
    plt.figure(num=fig_n)
    # fig.add_subplot(2, 2, p)
    plt.subplot(p)
    plt.imshow(I, cmap="gray")
    plt.title(name)
    plt.xticks([])
    plt.yticks([])


def histogram(img):
    height = img.shape[0]
    width = img.shape[1]

    hist = np.zeros((256))

    for i in np.arange(height):
        for j in np.arange(width):
            hist[img[i, j]] += 1

    return hist


def cumulative_histogram(hist):
    cum_hist = hist.copy()

    for i in np.arange(1, 256):
        cum_hist[i] = cum_hist[i-1] + cum_hist[i]

    return cum_hist


def equalize(I):
    h, w = I.shape
    N = h*w
    equ = np.zeros(I.shape)

    hist = histogram(I)
    cum = cumulative_histogram(hist)  # 需除以 N 才是 cdf

    for i in range(h):
        for j in range(w):
            # x = I[i, j]
            equ[i, j] = round(cum[I[i, j]]*255.0/N)
    return equ


def matching(I, I_ref):
    h, w = I.shape
    N = h*w
    h_r, w_r = I_ref.shape
    N_r = h_r*w_r

    # 取得兩圖的 gray scale histogram
    hist = histogram(I)
    hist_r = histogram(I_ref)

    # 作累積
    cum_hist = cumulative_histogram(hist)
    cum_hist_r = cumulative_histogram(hist_r)

    # cdf
    cdf = cum_hist/N
    cdf_r = cum_hist_r/N_r

    '''
    Transformation function
    z_k = G^{-1}[T(r)] => z_k = G^{-1}(s_k)
    k = 0,1,...,255
    G(z) - s >= 0
    '''
    K = 256
    new_val = np.zeros(K)
    for r in range(K):
        z = K - 1
        while z >= 0 and cdf[r] < cdf_r[z]:
            z -= 1
        new_val[r] = z

    I_new = np.zeros(I.shape)
    for i in range(h):
        for j in range(w):
            I_new[i, j] = new_val[I[i, j]]

    return I_new


def main():
    # Power-Law
    gammas = [0.04, 0.4, 1.0, 5.0]
    for i in range(4):
        new_I = (np.power(In/255, gammas[i]))
        set_image(new_I*255, int("22"+repr(i+1)), repr(gammas[i]), fig_num[0])
        cv2.imwrite(os.path.join(data_path, name+"_Powerlaw" +
                                 repr(gammas[i])+".jpg"), new_I*255)

    # Histogram Equalization
    equ1 = cv2.equalizeHist(In)
    equ2 = equalize(In)

    set_image(In, 131, "Origin", fig_num[1])
    set_image(equ1, 132, "Equalization1", fig_num[1])
    set_image(equ2, 133, "Equalization2", fig_num[1])
    cv2.imwrite(os.path.join(data_path, name+"_Equalization1"+".jpg"), equ1)
    cv2.imwrite(os.path.join(data_path, name+"_Equalization2"+".jpg"), equ2)

    # Histogram Matching
    ref = cv2.imread(os.path.join(
        data_path, "cat.jpg"), cv2.IMREAD_GRAYSCALE)
    mat = matching(In, ref)

    set_image(In, 131, "Origin", fig_num[2])
    set_image(ref, 132, "Ref", fig_num[2])
    set_image(mat, 133, "After matching", fig_num[2])
    cv2.imwrite(os.path.join(data_path, name+"_Matching"+".jpg"), mat)

    plt.show()


if __name__ == "__main__":
    main()
