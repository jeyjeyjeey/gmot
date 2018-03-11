# coding: utf-8
"""
lobi.play
ごまおつギルバト動画スクレイピング
"""

import os
import numpy as np
from progressbar import ProgressBar

import cv2
import syslog
from sqlalchemy.orm import sessionmaker

from gmot.data.DbAccessor import DBAccessor, GBPost
from gmot.ml.KNeighborsClassifierScikitLearn import knn_train
from gmot.gb.MvAnalyzer import ajust_capture, clip_caputure, extract_cap_total_score, extract_cap_end_score,\
                             KNN_IDENTIFIER_END_SCORE


MOVIE_DIR = '../movie'
TRAIN_DATA_DIR_END_SCORE = '../traindata/end_score'
TRAIN_DATA_DIR_TOTAL_SCORE = '../traindata/total_score'
TRAIN_DATA_DIR_MODE = '../traindata/mode'


def main():
    gb_posts_dict_list = get_target_data()

    # end_score[0-9]識別用のknnオブジェクトを生成
    # zero_to_ten = range(0, 10)
    # detect_chrs = np.append(zero_to_ten, '_')
    # knn_train(detect_chrs, TRAIN_DATA_DIR_END_SCORE, KNN_IDENTIFIER_END_SCORE, 3)

    gb_posts_dict_list = analyze_mv_file(gb_posts_dict_list, end_score=False, total_score=True)


def get_target_data():
    db = DBAccessor()
    db.prepare_connect(cf='../gb/config.ini')

    # 既存データ取得
    Session = sessionmaker(bind=db.engine)
    session = Session()
    gb_posts_result = (session.query(GBPost.id)
                       .filter(GBPost.is_valid_data == '0')
                       .order_by(GBPost.post_datetime.desc())
                       .offset(3000)
                       .limit(500)
                       # .all()
                       )
    session.flush()
    session.commit()

    gb_posts_dict_list = []
    for gb_post in gb_posts_result:
        gb_post_dict = gb_post._asdict()
        gb_posts_dict_list.append(gb_post_dict)
        
    return gb_posts_dict_list


def analyze_mv_file(gb_posts_dict_list, end_score=True, total_score=True):
    imgs_output_dir = 'cap'

    p = ProgressBar().start(len(gb_posts_dict_list), 1)
    for i, gb_posts_dict in enumerate(gb_posts_dict_list):
        id = gb_posts_dict['id']

        mv_name = gb_posts_dict['id'] + '.mp4'
        mv = cv2.VideoCapture(os.path.join(MOVIE_DIR, mv_name))

        # 1.end_score
        if end_score:
            proc_imgs = extract_cap_end_score(mv)
            if proc_imgs is None or len(proc_imgs) == 0:
                print('analyzeMvFile:end_score/InvalidMovie(This movie file has no frame): %s'
                      % mv_name)
                continue
            proc_imgs = clip_caputure(proc_imgs)
            proc_imgs = ajust_capture(proc_imgs)
            cv2.imwrite(os.path.join(
                imgs_output_dir, 'end_score_cap_%s.png' % (id)),
                proc_imgs[0])
            # _ = ocr_end_score_knn(proc_imgs, gb_posts_dict['id'], "end_score")

        # 2.total_score
        if total_score:
            proc_imgs = extract_cap_total_score(mv)
            if proc_imgs is None or len(proc_imgs) == 0:
                print('analyzeMvFile:end_score/InvalidMovie(This movie file has no frame): %s'
                      % mv_name)
                continue
            proc_imgs = clip_caputure(proc_imgs)
            proc_imgs = ajust_capture(proc_imgs)
            cv2.imwrite(os.path.join(
                imgs_output_dir, 'total_score_cap_%s.png' % (id)),
                proc_imgs[9])

        p.update(i + 1)

    return gb_posts_dict_list


if __name__ == '__main__':
    main()
