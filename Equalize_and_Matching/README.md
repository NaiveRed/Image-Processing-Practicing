# Equalize and Matching

練習 Power Law, Histogram Equalization, Histogram Matching。  
(Image Processing Homework1)  

P.S. 以下皆是將圖片轉為灰階後處理

## Power Law

直接對圖片數據做指數運算即可，這邊利用 4 個不同 gamma 值來比較。(在最後輸出圖片要再乘回 255)

`new_I = (np.power(In/255, gammas[i]))`

## Histogram Equalization

可直接使用 `equ1 = cv2.equalizeHist(In)`。  

也能先統計圖片的灰階值(histogram)，並一一做累積(cumulative)方便待會計算 cdf。  
接著進行 equalize，透過 cumulative/(HxW) 得到 cdf，再乘上 255 並四捨五入至整數後即是轉換的值。
(也就是 transformation function)

## Histogram Matching

(這邊不是很確定，有待確認...)

先對 input(欲修改的) 和 ref(目標) 求出 histogram，再分別對各灰階作累積。  
之後分別除以總 pixel 數就是 cdf 了。  
有了兩者的 cdf 後，就可用 input(r) 的對到其對應的 s。  
由 cdf 求，過程中可省略，直接用 cdf 值去作比較，再用 s 對回 ref(G(z)) 的。

## Reference

[doc. - histogram_equalization](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)  
https://github.com/machinelearninggod/Image-Processing-Algorithms
