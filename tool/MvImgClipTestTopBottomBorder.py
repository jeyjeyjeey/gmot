# coding: utf-8
"""
lobi.play
ごまおつギルバト動画スクレイピング
"""

import os
import numpy as np
from PIL import Image

import cv2
import syslog
from sqlalchemy.orm import sessionmaker

import gmot.data.DbAccessor as DbAccessor
from gmot.ml.CNNClassifierDigit import CNNClassifierDigit
from gmot.gb.MvAnalyzer import extract_cap_total_score, ajust_capture, \
    ocr_total_score, ocr_total_score_cnn, extract_cap_end_score, ocr_end_score, ocr_end_score_cnn,\
    KNN_IDENTIFIER_TOTAL_SCORE
from gmot.ml.KNeighborsClassifierScikitLearn import knn_train

MOVIE_DIR = '../movie'


def main():

    gb_posts_dict_list = get_target_data()

    gb_posts_dict_list = analyze_mv_file(gb_posts_dict_list)
    # update_record_with_user_id(gb_posts_dict_list)


def get_target_data():
    # 既存データ取得
    Session = sessionmaker(bind=DbAccessor.engine)
    session = Session()
    gb_posts_result = (session.query(DbAccessor.GBPost.id,
                                     DbAccessor.GBPost.is_valid_data
                                     )
                       .filter(DbAccessor.GBPost.id == '001633cf53753fa240eda995ad5fe9bb9810b8d5')
                       .limit(5)
                       # .offset(7000)
                       # .all()
                       )
    session.flush()
    session.commit()

    gb_posts_dict_list = []
    for gb_post in gb_posts_result:
        gb_post_dict = gb_post._asdict()
        gb_posts_dict_list.append(gb_post_dict)
        
    return gb_posts_dict_list


def analyze_mv_file(gb_posts_dict_list):
    
    print('analyzeMvFile/対象件数：' + str(len(gb_posts_dict_list)))

    remove_count = 0
    for i, gb_posts_dict in enumerate(gb_posts_dict_list):

        # print('before:')
        # print(gb_posts_dict)

        mv_name = gb_posts_dict['id'] + '.mp4'
        mv = cv2.VideoCapture(os.path.join(MOVIE_DIR, mv_name))

        proc_imgs = extract_cap_end_score(mv)
        if proc_imgs is None or len(proc_imgs) == 0:
            print('analyzeMvFile:end_score/InvalidMovie(This movie file has no frame): %s'
                  % mv_name)
            continue

        # Image.fromarray(np.uint8(proc_imgs[0])).show()

        proc_imgs = clip_caputure(proc_imgs)

        # Image.fromarray(np.uint8(proc_imgs[0])).show()

        proc_imgs = ajust_capture(proc_imgs)

        Image.fromarray(np.uint8(proc_imgs[0])).show()

        print('after:')
        print(gb_posts_dict)
        print('analyzeMvFile/処理完了：' + str(i))

    return gb_posts_dict_list


def clip_caputure(imgs):
    target_img = imgs[0]
    height, width = target_img.shape[:2]

    top_y = search_band_border(target_img, 0, lambda y: y + 1)
    bottom_y = search_band_border(target_img, height - 1, lambda y: y - 1)

    for i, img in enumerate(imgs):
        imgs[i] = img[top_y:bottom_y, :]

    return imgs


def search_band_border(img, y, dir_func):
    while True:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        # 色相 0-255
        # 色相環右回り　256変換されているのに注意
        hue = img_hsv.item(y, 0, 0)
        # 彩度 0-255
        # 大きくなるほど鮮やか
        saturation = img_hsv.item(y, 0, 1)
        # 明度 0-255
        # 大きくなるほど、明るい
        lightness = img_hsv.item(y, 0, 2)

        if saturation > 10 and lightness > 10:
            break

        y = dir_func(y)

    return y


def update_record_with_user_id(gb_posts_dict_list):

    # 既存データ取得
    Session = sessionmaker(bind=DbAccessor.engine)
    session = Session()

    # Mapping生成
    # [{
    #     'id': 1,
    #     'userId': 'hoge@hoge.com'
    # }, ...]
    gb_post_mappings = gb_posts_dict_list

    session.bulk_update_mappings(DbAccessor.GBPost, gb_post_mappings)
    session.flush()

    try:
        session.commit()
    except Exception as e:
        reason = str(e)
        syslog.syslog(reason)

    return True


if __name__ == '__main__':
    main()
