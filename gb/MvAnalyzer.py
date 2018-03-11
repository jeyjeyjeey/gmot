# coding: utf-8
"""
Analyze gmot guild battle movie
"""

import cv2
import os
import math
import numpy as np
import collections
import logging
import functools
from PIL import Image

from mlsp.ml.KNeighborsClassifierScikitLearn import knn_classify
from mlsp.ml.CNNClassifierDigit import CNNClassifierDigit
from mlsp.ml.RandomForestClassifierImage import RandomForestClassifierImage

KNN_IDENTIFIER_END_SCORE = 'end_score'
KNN_IDENTIFIER_TOTAL_SCORE = 'total_score'
KNN_IDENTIFIER_MODE = 'mode'

PREDICTION_MOST_SIGNIFICANT = 0.95
PREDICTION_ES_UNIDENTIFIED = 0.35
PREDICTION_TS_UNIDENTIFIED = 0.5

logger = logging.getLogger(__name__)


def extract_cap_end_score(mv):

    proc_imgs = []
    valid_last_frame = get_valid_last_frame(mv)

    if valid_last_frame:
        mv.set(cv2.CAP_PROP_POS_FRAMES, valid_last_frame)
        ret, frame = mv.read()
        proc_imgs.append(frame)
    else:
        return proc_imgs

    return proc_imgs


def extract_cap_total_score(mv):

    fps = mv.get(cv2.CAP_PROP_FPS)
    cnt_frame = mv.get(cv2.CAP_PROP_FRAME_COUNT)

    proc_imgs = []
    valid_last_frame = get_valid_last_frame(mv)
    if valid_last_frame:
        pos_frame = valid_last_frame
    else:
        return proc_imgs

    to_frame = pos_frame - fps * 10
    i = 0
    ret = True
    # check valid frame by certain interval(fps / 2) from last frame
    while ret and to_frame <= pos_frame:
        mv.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)

        ret, frame = mv.read()
        if ret is True:
            proc_imgs.append(frame)
        elif ret is False:
            break

        pos_frame -= fps / 2
        i += 1

    return proc_imgs if ret is True else None


def extract_cap_mode(mv, frame_rate=2):

    fps = mv.get(cv2.CAP_PROP_FPS)
    cnt_frame = mv.get(cv2.CAP_PROP_FRAME_COUNT)

    proc_imgs = []
    if cnt_frame > 180 * fps:
        pos_frame = fps * 8 + 1
    else:
        pos_frame = fps * 6 + 1

    i = 0
    ret = True
    # check valid frame by certain interval(fps) from last frame
    while ret and pos_frame > 0:
        mv.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)

        ret, frame = mv.read()
        if ret is True:
            proc_imgs.append(frame)
        elif ret is False:
            break

        pos_frame -= fps / frame_rate
        i += 1

    return proc_imgs if ret is True else None


def get_valid_last_frame(mv):

    fps = mv.get(cv2.CAP_PROP_FPS)
    cnt_frame = mv.get(cv2.CAP_PROP_FRAME_COUNT)

    # check valid frame by certain interval(fps) from last frame
    pos_frame = cnt_frame

    ret = False
    while not ret:
        mv.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)

        ret, frame = mv.read()
        if ret is True:
            break
        elif ret is False:
            pos_frame -= fps
            if pos_frame > 0:
                pass
            else:
                break

    return pos_frame if ret is True else False


def clip_caputure(imgs):
    target_img = imgs[0]
    height, width = target_img.shape[:2]

    start_x = math.floor((width / 2) - (width / 10))
    search_range_top = math.floor(height / 20)
    search_range_bottom = math.floor(height - (height / 20))

    top_y = search_band_border(target_img, start_x, 0,
                               search_range_top, 1)
    bottom_y = search_band_border(target_img, start_x, height - 1,
                                  search_range_bottom, -1)
    for i, img in enumerate(imgs):
        imgs[i] = img[top_y:bottom_y, :]

    return imgs


def search_band_border(img, x, y, search_range, incremental):

    while y * incremental < search_range * incremental:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        # 0-255 Hue circle (turn right)
        # attention:convert 360 to 256
        hue = img_hsv.item(y, x, 0)
        # 0-255 brighter as it gets larger
        saturation = img_hsv.item(y, x, 1)
        # 0-255 lighter as it gets larger
        lightness = img_hsv.item(y, x, 2)

        logger.debug("hue:%s, sat:%s, ltn:%s, inc:%s" % (hue, saturation, lightness, incremental))

        if lightness > 35:
            break

        y = y + incremental

    return y


def ajust_capture(imgs):

    if len(imgs) == 0:
        logger.error('ajustCapture/arg_imgs contains no imgs')
        return False

    # 1.775 : 568 / 320 : aspect ratio of iPhone gmot
    height_gmot = 568
    width_gmot = 320
    aspect_gmot = height_gmot / width_gmot 

    # Assumed that the first image has the same aspect ratio with others
    height, width = imgs[0].shape[:2] 
    height = math.floor(height)
    width = math.floor(width)
    aspect = height / width

    if aspect != aspect_gmot:
        # Find the appropriate width (correction width) for height
        coefficient_width = height / (aspect_gmot * width)
        correction_width = math.floor(width * coefficient_width)

        # Find the points x1, x2 where image is trimmed
        half_width = math.floor(width / 2)  # Center with respect to the X-axis
        x1 = int(half_width - (correction_width / 2))
        x2 = int(half_width + (correction_width / 2))

        # Trim a rectangular part passing through two points
        # img[y: y + h, x: x + w]
        for i, img in enumerate(imgs):
            imgs[i] = img[0:height, x1:x2]

    if not (height == height_gmot and width == width_gmot):
        for j, img in enumerate(imgs):
            imgs[j] = cv2.resize(img, (width_gmot, height_gmot))
        
    return imgs


def ocr_end_score_knn(imgs, id=None, imgs_output_dir=None):

    if len(imgs) > 1:
        logger.error('ocrEndScore/arg_imgs can only have a image:%s' % str(len(imgs)))
        return False
    elif len(imgs) == 0:
        logger.error('ocrEndScore/arg_imgs contains no imgs')
        return False
    img = imgs[0]

    digit_imgs = sample_end_score(img)

    # OCR KNN
    end_score_raw = knn_classify(digit_imgs, KNN_IDENTIFIER_END_SCORE)
    if end_score_raw is None:
        return None
    end_score = int(end_score_raw.replace('_', '0'))
    # img_output on
    if id is not None and imgs_output_dir is not None:
        for num, knn_digit in enumerate(end_score_raw):
            cv2.imwrite(os.path.join(
                imgs_output_dir, knn_digit,
                'end_score_digit_%s_%s.png' % (knn_digit, id)
            ),
                digit_imgs[num])

    return end_score_raw, end_score


def ocr_end_score_cnn(imgs, cnn: CNNClassifierDigit, id=None, imgs_output_dir=None):

    if len(imgs) > 1:
        logger.error('ocrEndScore/arg_imgs can only have a image:%s' % str(len(imgs)))
        return False
    elif len(imgs) == 0:
        logger.error('ocrEndScore/arg_imgs contains no imgs')
        return False
    img = imgs[0]

    digit_imgs = sample_end_score(img)

    # OCR CNN
    end_score_raw, prediction_list = cnn.classify(digit_imgs, prediction_filtering=False)
    if end_score_raw is None:
        return None

    # Filtering with prediction
    end_score_raw_filtering = ''
    for i, end_score_raw_digit in enumerate(end_score_raw):
        if i == 0:
            if end_score_raw_digit not in ('0', '1'):
                end_score_raw_filtering += '_'
            elif prediction_list[i] < PREDICTION_MOST_SIGNIFICANT and end_score_raw_digit in ('0', '1'):
                end_score_raw_filtering += '0'
            else:
                end_score_raw_filtering += end_score_raw_digit

        if i != 0:
            if prediction_list[i] < PREDICTION_ES_UNIDENTIFIED:
                end_score_raw_filtering += '_'
            else:
                end_score_raw_filtering += end_score_raw_digit

    end_score = int(end_score_raw_filtering.replace('_', '0'))
    # img_output on
    if id is not None and imgs_output_dir is not None:
        for num, knn_digit in enumerate(end_score_raw):
            cv2.imwrite(os.path.join(
                imgs_output_dir, knn_digit,
                'end_score_digit_%s_%s.png' % (knn_digit, id)
            ),
                digit_imgs[num])

    return end_score_raw_filtering, end_score, prediction_list


def sample_end_score(img):
    img = img[71:91, 35:135]        # Extract letter area　rectangular[y: y + h, x: x + w]
    img = conv_gray_img(img)        # Convert to gray scale
    img = adpt_thresh_img(img, 25)  # Adaptive thresholding
    # Extract each digit area x:15+15+15+3(comma)+15+15+15
    digit_imgs = [
        img[0:20, 8:22],
        img[0:20, 23:37],
        img[0:20, 38:52],
        img[0:20, 56:70],
        img[0:20, 71:85],
        img[0:20, 86:100]
    ]

    return digit_imgs


def ocr_total_score_knn(imgs, id=None, imgs_output_dir=None):

    if len(imgs) == 0:
        logger.error('ocrTotalScore/arg_imgs contains no imgs')
        return False

    total_results = []
    total_score_raw_all = []
    for i, img in enumerate(imgs):
        digit_imgs = sample_total_score(img)

        # OCR KNN
        total_score_raw = knn_classify(digit_imgs, KNN_IDENTIFIER_TOTAL_SCORE)
        if total_score_raw is None:
            return None
        elif total_score_raw.count('_') <= 1:
            total_results.append(total_score_raw)
            # img_output on
            if id is not None and imgs_output_dir is not None:
                for num, digit in enumerate(total_score_raw):
                    cv2.imwrite(os.path.join(
                                    imgs_output_dir, digit,
                                    'total_score_digit_%s_%s.png' % (digit, id)
                                    ),
                                digit_imgs[num])
        total_score_raw_all.append(total_score_raw)

    if len(total_results) != 0:
        count_dict = collections.Counter(total_results)
        total_score = int(count_dict.most_common(1)[0][0].replace('_', '0'))
        total_score_count = int(count_dict.most_common(1)[0][1])
    else:
        total_score = 0
        total_score_count = 0

    return total_score, collections.Counter(total_score_raw_all), total_score_count


def ocr_total_score_cnn(imgs, cnn: CNNClassifierDigit, id=None, imgs_output_dir=None):

    if len(imgs) == 0:
        logger.error('ocrTotalScore/arg_imgs contains no imgs')
        return False

    total_results_score = []
    total_results_prediction = []
    total_score_raw_all = []
    for img in imgs:
        digit_imgs = sample_total_score(img)

        # OCR CNN
        total_score_raw, prediction_list = cnn.classify(digit_imgs, prediction_filtering=False)
        if total_score_raw is None:
            return None

        # Filtering with prediction
        end_score_raw_filtering = ''
        for i, total_score_raw_digit in enumerate(total_score_raw):
            if i == 0:
                if total_score_raw_digit not in ('0', '1'):
                    end_score_raw_filtering += '0'
                elif prediction_list[i] < PREDICTION_MOST_SIGNIFICANT and total_score_raw_digit in ('0', '1'):
                    end_score_raw_filtering += '_'
                else:
                    end_score_raw_filtering += total_score_raw_digit

            if i != 0:
                if prediction_list[i] < PREDICTION_TS_UNIDENTIFIED:
                    end_score_raw_filtering += '_'
                else:
                    end_score_raw_filtering += total_score_raw_digit

        if (end_score_raw_filtering.count('_') == 0 or
           (end_score_raw_filtering.count('_') == 1 and end_score_raw_filtering[0] == '_')):
            total_results_score.append(end_score_raw_filtering)
            total_results_prediction.append(prediction_list)
            # img_output on
            if id is not None and imgs_output_dir is not None:
                for num, digit in enumerate(end_score_raw_filtering):
                    cv2.imwrite(os.path.join(
                                    imgs_output_dir, digit,
                                    'total_score_cnn_digit_%s_%s.png' % (digit, id)
                                    ),
                                digit_imgs[num])
        total_score_raw_all.append(end_score_raw_filtering)

    if len(total_results_score) != 0:
        count_dict = collections.Counter(total_results_score)
        total_score_str = count_dict.most_common(1)[0][0]
        total_score = int(total_score_str.replace('_', '0'))
        total_score_count = int(count_dict.most_common(1)[0][1])
        most_common_prediction_list = []
        for where_i in [si for si, s in enumerate(total_results_score) if s == total_score_str]:
            most_common_prediction_list.append(total_results_prediction[where_i])
        prediction_list_mean = np.array(most_common_prediction_list).mean(axis=0).tolist()
    else:
        total_score = 0
        total_score_str = ''
        total_score_count = 0
        prediction_list_mean = []

    return total_score, total_score_str, total_score_count, prediction_list_mean


def sample_total_score(img):
    img = img[415:445, 163:281]     # Extract letter area　rectangular[y: y + h, x: x + w]
    img = conv_gray_img(img)        # Convert to gray scale
    img = adpt_thresh_img(img, 25)  # Adaptive thresholding
    # Extract each digit area x:15+15+15+3(comma)+15+15+15
    digit_imgs = [
        img[0:30, 0:19],
        img[0:30, 19:38],
        img[0:30, 38:57],
        img[0:30, 61:80],
        img[0:30, 80:99],
        img[0:30, 99:118]
    ]
    return digit_imgs


def discern_mode(imgs, id=None, imgs_output_dir=None):
        
    if len(imgs) == 0:
        logger.error('discernMode/arg_imgs contains no imgs')
        return False

    knn_stage_mode_results = []
    break_images = sample_discern_mode(imgs)

    # DISCERN KNN
    stage_mode_raw = knn_classify(break_images, KNN_IDENTIFIER_MODE)
    if stage_mode_raw is None:
        return None
    # img_output on
    logger.debug(stage_mode_raw)
    # [Image.fromarray(np.uint8(break_image)).show() for break_image in break_images]
    if id is not None and imgs_output_dir is not None:
        for num, knn_digit in enumerate(stage_mode_raw):
            cv2.imwrite(os.path.join(
                imgs_output_dir, knn_digit,
                'mode_%s_%s.png' % (knn_digit, id)
            ),
                break_images[num])

    # When even one is determined the Break, set the Break
    stage_mode = 'b' if 'b' in stage_mode_raw else 'n'

    return stage_mode


def discern_mode_rf(imgs, clf: RandomForestClassifierImage, id=None, imgs_output_dir=None):
    if len(imgs) == 0:
        logger.error('discernMode/arg_imgs contains no imgs')
        return False

    break_images = sample_discern_mode(imgs)
    break_images = clf.sample_image(
                    break_images,
                    normalization=True,
                    flattening=True)

    # DISCERN KNN
    stage_mode_raw = clf.classify(break_images)
    if stage_mode_raw is None:
        return None
    # img_output on
    logger.debug(stage_mode_raw)
    # [Image.fromarray(np.uint8(break_image)).show() for break_image in break_images]
    if id is not None and imgs_output_dir is not None:
        for num, knn_digit in enumerate(stage_mode_raw):
            cv2.imwrite(os.path.join(
                imgs_output_dir, knn_digit,
                'mode_%s_%s.png' % (knn_digit, id)
            ),
                break_images[num])

    # When even one is determined the Break, set the Break
    stage_mode = 'n'
    if len(np.where(stage_mode_raw == 'b')[0]):
        indices_b = np.argwhere(stage_mode_raw == 'b')
        if (indices_b.max() - indices_b.min()) == (indices_b.size - 1):
            stage_mode = 'b'

    return stage_mode


def discern_mode_cnn(imgs, cnn, id=None, imgs_output_dir=None):
    if len(imgs) == 0:
        logger.error('discernMode/arg_imgs contains no imgs')
        return False

    break_images = sample_discern_mode(imgs)
    break_images = np.array(
        cnn.sample_image(
            break_images,
            resized_shape=(cnn.input_x, cnn.input_y),
            # normalization=True,
            other_sample_func_list=[
                functools.partial(
                    cnn.convert_channels,
                    mode=cv2.COLOR_GRAY2BGR),
                cnn.normalize
            ])
    )

    # DISCERN CNN
    stage_mode_raw, predictions = cnn.classify(break_images)
    if stage_mode_raw is None:
        return None
    # img_output on
    logger.debug(stage_mode_raw)
    # [Image.fromarray(np.uint8(break_image)).show() for break_image in break_images]
    if id is not None and imgs_output_dir is not None:
        for num, knn_digit in enumerate(stage_mode_raw):
            cv2.imwrite(os.path.join(
                imgs_output_dir, knn_digit,
                'mode_%s_%s.png' % (knn_digit, id)
            ),
                break_images[num])

    # When even one is determined the Break, set the Break
    stage_mode = 'n'
    for i in range(len(stage_mode_raw)):
        if np.max(predictions[i]) < 0.95:
            continue
        elif stage_mode_raw[i] == 'b':
            stage_mode = 'b'
            break

    return stage_mode, predictions


def sample_discern_mode(imgs):
    for i, img in enumerate(imgs):
        img = img[195:250, 40:280]  # Extract letter area　rectangular[y: y + h, x: x + w]
        img = conv_gray_img(img)  # Convert to gray scale
        img = adpt_thresh_img(img, 25)  # Adaptive thresholding
        imgs[i] = img

    return imgs


def conv_gray_img(img):
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    return img


def adpt_thresh_img(img, thresh_substract_const=25, block_size=11):
    max_pixel = 255
    img = cv2.adaptiveThreshold(
        img,
        max_pixel,
        cv2.ADAPTIVE_THRESH_MEAN_C,   # The median value of the neighboring region is set as a threshold value
        cv2.THRESH_BINARY,
        block_size,                   # Size of eighboring region used for threshold calculation
        thresh_substract_const        # A constant that subtracts the calculated threshold value
    )

    return img


def sharpen_img(img):
    k = 1.0
    op = np.array([
                    [-k, -k, -k],
                    [-k, 1 + 8 * k, -k],
                    [-k, -k, -k]
                ])
    img = cv2.filter2D(img, -1, op)
    img = cv2.convertScaleAbs(img)

    return img


def put_text(img, text, x, y, font_size=0.6, color=(255, 255, 255), font=cv2.FONT_HERSHEY_PLAIN):
    cv2.putText(img, text, (x, y), font, font_size, color)

    return img
