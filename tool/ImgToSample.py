import cv2
import numpy as np

#画像取得
imgPath = './capture/f6c8cd017793b8b94f9244f623afbe3f3682bb15_scrF.png'
img = cv2.imread(imgPath)

#文字領域抽出　矩形
#2点を通る矩形部分を切り抜き
#img[y: y + h, x: x + w]
# img = img[71:91, 35:135]
img = img[415:445, 160:280]

#グレースケール変換
img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

k = 0.2
op = np.array([
                [-k, -k, -k],
                [-k, 1 + 8 * k, -k],
                [-k, -k, -k]
            ])
img = cv2.filter2D(img, -1, op)
img = cv2.convertScaleAbs(img)

# #適応的二値変換
# #ADAPTIVE_THRESH_MEAN_C 近傍領域の中央値をしきい値
# #ADAPTIVE_THRESH_GAUSSIAN_C 近傍領域の重み付け平均値をしきい値
max_pixel = 255
blockSize = 11               #しきい値計算に使用する近傍領域のサイズ
threshSubstractConst = 25    #計算されたしきい値から引く定数
img = cv2.adaptiveThreshold(
    img,
    max_pixel,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    blockSize,
    threshSubstractConst
)



cv2.imwrite('./3s.png', img)
