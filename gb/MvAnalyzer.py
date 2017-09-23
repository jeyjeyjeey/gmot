# coding: utf-8
"""
lobi.play
ごまおつギルバト動画解析
"""

import cv2
import math
import numpy as np
import collections

from gmot.ml.KNeighborsClassifierScikitLearn import knnClassify

KNN_IDENTIFIER_END_SCORE = 'end_score'
KNN_IDENTIFIER_TOTAL_SCORE = 'total_score'
KNN_IDENTIFIER_MODE = 'mode'

def extractCapEndScore(mv):

    proc_imgs = []
    # 有効な最終フレームを取得する
    valid_last_frame = getValidLastFrame(mv)

    if valid_last_frame:
        mv.set(cv2.CAP_PROP_POS_FRAMES, valid_last_frame)
        ret, frame = mv.read()
        proc_imgs.append(frame)
    else:
        return proc_imgs

    return proc_imgs

def extractCapTotalScore(mv):

    # 動画情報を取得
    fps = mv.get(cv2.CAP_PROP_FPS)
    cnt_frame = mv.get(cv2.CAP_PROP_FRAME_COUNT)

    proc_imgs = []
    # 有効な最終フレームを取得する
    valid_last_frame = getValidLastFrame(mv)
    if valid_last_frame:
        pos_frame = valid_last_frame
    else:
        return proc_imgs
    
    # 現在フレームは1.で取得した有効な最終フレームから開始
    # 末尾から15フレーム（0.5秒）毎に取得する
    to_frame = pos_frame - fps * 10

    i = 0
    ret = True
    # 最終フレームから一定間隔(fps)ごとに有効な画像を確認していく
    while ret and to_frame <= pos_frame:
        mv.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)

        ret, frame = mv.read()
        if ret == True:
            proc_imgs.append(frame)
        elif ret == False:
            break

        pos_frame -= fps / 2
        i += 1

    return proc_imgs if ret == True else None

def extractCapMode(mv):

    # 動画情報を取得
    fps = mv.get(cv2.CAP_PROP_FPS)
    cnt_frame = mv.get(cv2.CAP_PROP_FRAME_COUNT)

    proc_imgs = []
    if cnt_frame > 180 * fps:
        pos_frame = fps * 8 + 1
    else:
        pos_frame = fps * 6 + 1

    i = 0
    ret = True
    # 最終フレームから一定間隔(fps)ごとに有効な画像を確認していく
    while ret and pos_frame > 0:
        mv.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)

        ret, frame = mv.read()
        if ret == True:
            proc_imgs.append(frame)
        elif ret == False:
            break

        pos_frame -= fps / 2
        i += 1

    return proc_imgs if ret == True else None

def getValidLastFrame(mv):

    # 動画情報を取得
    fps = mv.get(cv2.CAP_PROP_FPS)
    cnt_frame = mv.get(cv2.CAP_PROP_FRAME_COUNT)

    #最終フレームから一定間隔(fps)ごとに確認し、有効な最終フレームを取得する
    pos_frame = cnt_frame

    ret = False
    while not ret:
        mv.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)

        ret, frame = mv.read()
        if ret == True:
            break
        elif ret == False:
            pos_frame -= fps
            if pos_frame > 0:
                pass
            else:
                break

    return pos_frame if ret == True else False

def ajustCapture(imgs):

    if len(imgs) == 0:
        print('ajustCapture/画像ファイルがないよ')
        return False

    # 1.775 : 568 / 320 iPhone ごま乙の画面サイズ比
    height_gmot = 568
    width_gmot = 320
    aspect_gmot = height_gmot / width_gmot 

    # 一枚目の画像が他の画像と同一のアスペクト・サイズであることが前提
    height, width = imgs[0].shape[:2] 
    height = math.floor(height)
    width = math.floor(width)
    aspect = height / width

    # アスペクト比調整
    if  aspect != aspect_gmot:
        #高さに対する適正な幅（補正幅）を求める
        coefficientWidth = height / (aspect_gmot * width) 
        correctionWidth = math.floor(width * coefficientWidth)

        #トリミングする位置x1,x2を求める
        halfWidth = math.floor(width / 2) #x軸中央位置
        x1 = int(halfWidth - (correctionWidth / 2))
        x2 = int(halfWidth + (correctionWidth / 2))

        # 2点を通る矩形部分を切り抜き
        # img[y: y + h, x: x + w]
        for i, img in enumerate(imgs):
            imgs[i] = img[0:height, x1:x2]

    # 画像サイズ調整
    if not (height == height_gmot and width == width_gmot):
        for j, img in enumerate(imgs):
            imgs[j] = cv2.resize(img, (width_gmot, height_gmot))
        
    return imgs

def ocrEndScore(imgs):

    if len(imgs) > 1:
        print('ocrEndScore/画像ファイルは1枚のみの想定:%s' % str(len(imgs)))
        return False
    elif len(imgs) == 0:
        print('ocrEndScore/画像ファイルがないよ')
        return False
    img = imgs[0]

    # 画像処理
    img = img[71:91, 35:135]        # 文字領域抽出　矩形[y: y + h, x: x + w]
    img = convGrayImg(img)          # グレースケール変換
    img = adptThreshImg(img, 25)    # 適応的二値変換
    # 各数値領域抽出 x:15+15+15+3(カンマ)+15+15+15
    digit_imgs = [
        img[0:20, 8:22],
        img[0:20, 23:37],
        img[0:20, 38:52],
        img[0:20, 56:70],
        img[0:20, 71:85],
        img[0:20, 86:100]
    ]

    # OCR KNN
    end_score_raw = knnClassify(digit_imgs, KNN_IDENTIFIER_END_SCORE)
    end_score = int(end_score_raw.replace('_', '0'))

    # debug
    # #digitImg output
    # for num, digitImg in enumerate(digitImgs):
    #     digitImgName = str(num) + '_' + 何かID +'.png'
    #     digitImgPath = os.path.join(CAPTURE_DIR, digitImgName)
    #     cv2.imwrite(digitImgPath, digitImg)

    return end_score_raw, end_score

def ocrTotalScore(imgs):

    if len(imgs) == 0:
        print('ocrTotalScore/画像ファイルがないよ')
        return False

    knn_total_results = []
    total_score_raw_all = []
    for i, img in enumerate(imgs):
        # 画像処理
        img = img[415:445, 163:281]     # 文字領域抽出　矩形[y: y + h, x: x + w]
        img = convGrayImg(img)          # グレースケール変換
        img = adptThreshImg(img, 25)    # 適応的二値変換
        # 各数値領域抽出 x:19+19+19+4(カンマ)+19+19+19
        digit_imgs = [
            img[0:30, 0:19],
            img[0:30, 19:38],
            img[0:30, 38:57],
            img[0:30, 61:80],
            img[0:30, 80:99],
            img[0:30, 99:118]
        ]
        # debug
        # Img output
        # cv2.imwrite(os.path.join(CAPTURE_DIR, '%s_trm_scrT_%s.png' % (post_id, str(i))),
        #             img)
        # # digitImg output
        # for num, digit_img in enumerate(digit_imgs):
        #     cv2.imwrite(os.path.join(CAPTURE_DIR, '%s_scrT_digit_%s_%s.png' % (str(num), post_id, str(i))),
        #                 digit_img)

        # OCR KNN
        total_score_raw = knnClassify(digit_imgs, KNN_IDENTIFIER_TOTAL_SCORE)
        total_score_raw_all.append(total_score_raw)
        if total_score_raw.count('_') <= 1:
            knn_total_results.append(total_score_raw)

    len_result = len(knn_total_results)
    if len(knn_total_results) != 0:
        count_dict = collections.Counter(knn_total_results)
        total_score = int(count_dict.most_common(1)[0][0].replace('_', '0'))
        total_score_count = int(count_dict.most_common(1)[0][1])
    else:
        total_score = 0
        total_score_count = 0

    return total_score, collections.Counter(total_score_raw_all), total_score_count

def discernMode(imgs):
        
    if len(imgs) == 0:
        print('discernMode/画像ファイルがないよ')
        return False

    knn_mode_results = []
    break_images = []
    for i, img in enumerate(imgs):
        # 画像処理
        img = img[195:250, 40:280]      # 文字領域抽出　矩形[y: y + h, x: x + w]
        img = convGrayImg(img)          # グレースケール変換
        img = adptThreshImg(img, 25)    # 適応的二値変換
        break_images.append(img)

    # DISCERN KNN
    knn_mode_results.append(knnClassify(break_images, KNN_IDENTIFIER_MODE))

    # 一つでもブレイクと判別された場合、ブレイクと判定する
    stage_mode = 'b' if 'b' in knn_mode_results else 'n'

    # debug
    # print('id:' + post_detail.id)
    # print('stage_mode:' + post_detail_list[i].stage_mode)
    # for num, break_image in enumerate(break_images):
    #     break_image_name = str(num) + '_' + 何かID + '.png'
    #     break_image_path = os.path.join(CAPTURE_DIR, break_image_name)
    #     cv2.imwrite(break_image_path, break_image)

    return stage_mode 

def convGrayImg(img):
    # グレースケール変換
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    return img

def adptThreshImg(img, threshSubstractConst):
    # 適応的二値変換
    # ADAPTIVE_THRESH_MEAN_C 近傍領域の中央値をしきい値
    # ADAPTIVE_THRESH_GAUSSIAN_C 近傍領域の重み付け平均値をしきい値
    max_pixel = 255
    block_size = 11               
    thresh_substract_const = 25    
    img = cv2.adaptiveThreshold(
        img,
        max_pixel,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,                  #しきい値計算に使用する近傍領域のサイズ
        thresh_substract_const        #計算されたしきい値から引く定数
    )

    return img

def sharpenImg(img):
    k = 1.0
    op = np.array([
                    [-k, -k, -k],
                    [-k, 1 + 8 * k, -k],
                    [-k, -k, -k]
                ])
    img = cv2.filter2D(img, -1, op)
    img = cv2.convertScaleAbs(img)

    return img
