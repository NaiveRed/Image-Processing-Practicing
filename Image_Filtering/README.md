# Filter Practicing

利用 filter 將照片做幾種變化並比較。  
可粗略的針對偵測到的臉(方形區域)做處理。  
(Image Processing Homework2: Face picture with beauty treatment)

## Use

Python: 3.6  

`python image_filtering.py`

變數定義  
```
ONLY_FACE:
    True: Filter 只對臉的部分產生效果
    False: 對整張圖片進行處理
SHOW_ROI: (當 ONLY_FACE=True)
    True: 標示出所偵測且處理的臉的區域
    False: 不標示
```

## Method

使用不同的 kernel 來對同一張圖片做處理並比較差異。  
主要利用 `res = cv.filter2D(img, ddepth=cv.CV_8U, kernel=kernel)` 來完成。

* **Sharpen:**

    kernel:  
    ```
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
    ```

    圖片的顆粒感變得較重，邊緣變得更加銳利。

* **Gaussian blur 3x3:**

    kernel:  
    ```
    [[1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]]/16
    ```

    其消除雜訊的效果使的圖片整體平滑許多。  
    > Gaussian blurring is highly effective in removing gaussian noise from the image. - doc.

* **Bilateral filter:**

    直接利用 `res = cv.bilateralFilter(img, 5, 75, 75)`

    在消除雜訊的情況下也保持了一定的邊界清晰，看起來比 Gaussian blur 還好看。  
    > highly effective in noise removal while keeping edges sharp. - doc.
    
## Face Detection

    使用 opencv 內建的 Haar cascade classification 來偵測臉部。  
    可以針對所偵測到的區域進行處理，並框出來做比較。

## Reference

[doc. - Smoothing Images](https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html)  
[OpenCV-Python中文教程 - 图像平滑](https://www.kancloud.cn/aollo/aolloopencv/269599)  
[doc. - filter2D](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04)  
[WIKI. - kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing))

(測試圖片取自於網路)