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

import gmot.data.DbAccessor as DbAccessor
from gmot.ml.KNeighborsClassifierScikitLearn import knn_train
from gmot.gb.MvAnalyzer import extract_cap_total_score, ajust_capture, ocr_total_score,\
                             extract_cap_mode, discern_mode, extract_cap_end_score, ocr_end_score,\
                             KNN_IDENTIFIER_END_SCORE, KNN_IDENTIFIER_TOTAL_SCORE, KNN_IDENTIFIER_MODE


MOVIE_DIR = '../movie'
TRAIN_DATA_DIR_END_SCORE = '../traindata/end_score'
TRAIN_DATA_DIR_TOTAL_SCORE = '../traindata/total_score'
TRAIN_DATA_DIR_MODE = '../traindata/mode'


def main():
    gb_posts_dict_list = get_target_data()

    # end_score[0-9]識別用のknnオブジェクトを生成
    zero_to_ten = range(0, 10)
    detect_chrs = np.append(zero_to_ten, '_')
    knn_train(detect_chrs, TRAIN_DATA_DIR_END_SCORE, KNN_IDENTIFIER_END_SCORE, 3)
    # # total_score[0-9]識別用のknnオブジェクトを生成
    # detect_chrs = np.append(zero_to_ten, '_')
    # knn_train(detect_chrs, TRAIN_DATA_DIR_TOTAL_SCORE, KNN_IDENTIFIER_TOTAL_SCORE, 3)
    # # break,nobreak識別用のknnオブジェクトを生成
    # detect_chrs = ['b', 'n']
    # knn_train(detect_chrs, TRAIN_DATA_DIR_MODE, KNN_IDENTIFIER_MODE, 3)

    gb_posts_dict_list = analyze_mv_file(gb_posts_dict_list)


def get_target_data():
    # 既存データ取得
    Session = sessionmaker(bind=DbAccessor.engine)
    session = Session()
    gb_posts_result = (session.query(DbAccessor.GBPost.id)
                       .filter(DbAccessor.GBPost.is_valid_data == '0')
                       .order_by(DbAccessor.GBPost.post_datetime.desc())
                       .offset(2000)
                       .limit(2000)
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

    p = ProgressBar().start(len(gb_posts_dict_list), 1)
    for i, gb_posts_dict in enumerate(gb_posts_dict_list):

        mv_name = gb_posts_dict['id'] + '.mp4'
        mv = cv2.VideoCapture(os.path.join(MOVIE_DIR, mv_name))

        # 1.end_score
        proc_imgs = extract_cap_end_score(mv)
        if proc_imgs is None or len(proc_imgs) == 0:
            print('analyzeMvFile:end_score/InvalidMovie(This movie file has no frame): %s'
                  % mv_name)
            continue
        proc_imgs = ajust_capture(proc_imgs)
        _ = ocr_end_score(proc_imgs, gb_posts_dict['id'], "end_score")
        # # 2.total_score
        # proc_imgs = extract_cap_total_score(mv)
        # if proc_imgs is None or len(proc_imgs) == 0:
        #     print('analyzeMvFile:total_score/InvalidMovie(This movie file has no frame): %s'
        #           % mv_name)
        #     continue
        # proc_imgs = ajust_capture(proc_imgs)
        # _ = ocr_total_score(proc_imgs, gb_posts_dict['id'], "total_score")
        # # 3.break/nobreak
        # proc_imgs = extract_cap_mode(mv)
        # if proc_imgs is None or len(proc_imgs) == 0:
        #     print('analyzeCapture:mode/InvalidMovie(This movie file has no frame): %s'
        #           % mv_name)
        #     continue
        # proc_imgs = ajust_capture(proc_imgs)
        # _ = discern_mode(proc_imgs, gb_posts_dict['id'], "mode")

        p.update(i + 1)

    return gb_posts_dict_list


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
