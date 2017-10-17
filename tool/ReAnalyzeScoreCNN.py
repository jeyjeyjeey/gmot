# coding: utf-8
"""
lobi.play
ごまおつギルバト動画スクレイピング
"""

import os
import logging
import datetime
import numpy as np

import cv2
from PIL import Image
from sqlalchemy.orm import sessionmaker

import gmot.data.DbAccessor as DbAccessor
from gmot.ml.CNNClassifierDigit import CNNClassifierDigit
from gmot.gb.MvAnalyzer import extract_cap_total_score, ajust_capture, clip_caputure, put_text,\
    ocr_total_score_cnn, extract_cap_end_score, ocr_end_score_cnn,\
    PREDICTION_MOST_SIGNIFICANT, PREDICTION_ES_UNIDENTIFIED, PREDICTION_TS_UNIDENTIFIED
from gmot.ml.KNeighborsClassifierScikitLearn import knn_train

MOVIE_DIR = '../movie'
TRAIN_DATA_DIR_TOTAL_SCORE_KNN = '../traindata/total_score_knn'
TRAIN_DATA_DIR_TOTAL_SCORE_CNN = '../traindata/total_score_cnn'
CKPT_DATA_DIR_TOTAL_SCORE_CNN = '../ckpt/total_score'
TRAIN_DATA_DIR_END_SCORE_CNN = '../traindata/end_score_cnn'
CKPT_DATA_DIR_END_SCORE_CNN = '../ckpt/end_score'


def main():
    logging.basicConfig(filename='../log/%s_%s.log'
                                 % (os.path.basename(__file__), datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
                        level=logging.DEBUG, format='%(asctime)s %(message)s')

    gb_posts_dict_list = get_target_data()
    gb_posts_dict_list = analyze_mv_file(gb_posts_dict_list)


def get_target_data():
    # 既存データ取得
    Session = sessionmaker(bind=DbAccessor.engine)
    session = Session()
    gb_posts_result = (session.query(DbAccessor.GBPost.id,
                                     DbAccessor.GBPost.is_valid_data,
                                     DbAccessor.GBPost.end_score_raw,
                                     DbAccessor.GBPost.stage_mode
                                     )
                       # .filter(DbAccessor.GBPost.id.in_(('04cf12ef76bc8ee5de82247feeb024d54958d428',
                       #                                   '015d64fc7c555f8b1ee72d906833168a87fb68d5')))
                       .filter(DbAccessor.GBPost.id == '7e680ba8cc19dafb4f9b48460d5626c36c3f611e')
                       # .limit(100)
                       # .offset(1)
                       .all()
                       )
    session.flush()
    session.commit()

    gb_posts_dict_list = []
    for gb_post in gb_posts_result:
        gb_post_dict = gb_post._asdict()
        # if gb_post_dict['end_score_raw'] is None:
        gb_posts_dict_list.append(gb_post_dict)
        
    return gb_posts_dict_list


def analyze_mv_file(gb_posts_dict_list):

    logging.info('analyzeMvFile/対象件数：' + str(len(gb_posts_dict_list)))

    es_cnn = CNNClassifierDigit()
    es_cnn.set_train_data_dir(TRAIN_DATA_DIR_END_SCORE_CNN)
    es_cnn.set_ckpt_dir(CKPT_DATA_DIR_END_SCORE_CNN)
    es_cnn.prepare()
    es_cnn.run_train()

    ts_cnn = CNNClassifierDigit()
    ts_cnn.set_train_data_dir(TRAIN_DATA_DIR_TOTAL_SCORE_CNN)
    ts_cnn.set_ckpt_dir(CKPT_DATA_DIR_TOTAL_SCORE_CNN)
    ts_cnn.prepare()
    ts_cnn.run_train()

    gb_posts_dict_list_commit = []
    for i, gb_posts_dict in enumerate(gb_posts_dict_list):

        logging.info('before:')
        logging.info(gb_posts_dict)

        mv_name = gb_posts_dict['id'] + '.mp4'
        mv = cv2.VideoCapture(os.path.join(MOVIE_DIR, mv_name))
        fps = mv.get(cv2.CAP_PROP_FPS)
        cnt_frame = mv.get(cv2.CAP_PROP_FRAME_COUNT)
        stage_mode = gb_posts_dict['stage_mode']

        # 1.end_score
        proc_imgs = extract_cap_end_score(mv)
        if proc_imgs is None or len(proc_imgs) == 0:
            logging.error('analyzeMvFile:end_score/InvalidMovie(This movie file has no frame): %s'
                          % mv_name)
            continue
        proc_imgs = clip_caputure(proc_imgs)
        proc_imgs = ajust_capture(proc_imgs)
        # proc_imgs[0] = put_text(proc_imgs[0], gb_posts_dict['id'], 10, 10, 0.6)
        # Image.fromarray(np.uint8(proc_imgs[0])).show()
        end_score_raw, end_score, end_score_prediction = ocr_end_score_cnn(proc_imgs, es_cnn)

        # 2.total_score
        proc_imgs = extract_cap_total_score(mv)
        if proc_imgs is None or len(proc_imgs) == 0:
            logging.error('analyzeMvFile:total_score/InvalidMovie(This movie file has no frame): %s'
                          % mv_name)
            continue
        proc_imgs = clip_caputure(proc_imgs)
        proc_imgs = ajust_capture(proc_imgs)
        # proc_imgs[0] = put_text(proc_imgs[0], gb_posts_dict['id'], 10, 10, 0.6)
        # Image.fromarray(np.uint8(proc_imgs[0])).show()
        (total_score,
         total_score_raw,
         total_score_count,
         total_score_prediction) = ocr_total_score_cnn(proc_imgs, ts_cnn)

        # data_cleanse
        # final_scoreをセットする
        # TS/ES一方が全識別不能の場合、判定無しでもう一方を採用
        if total_score == 0:
            final_score = end_score
        elif end_score == 0:
            final_score = total_score
        # ES/TSの各桁の確率を比較し、高い数字を採用する
        else:
            cmpr_scr_list = np.array([end_score_raw, total_score_raw])
            cmpr_pred_list = np.array([end_score_prediction, total_score_prediction])
            if total_score_count <= 1:  # total_scoreの取得元フレームが少ない場合、信頼性を下げる
                cmpr_pred_list[1] = cmpr_pred_list[1] - 0.05
            max_pred_column_indices = np.argmax(cmpr_pred_list, axis=0).reshape(-1).tolist()
            final_score_raw = str()
            for which_digit, which_score in enumerate(max_pred_column_indices):
                final_score_raw += cmpr_scr_list[which_score][which_digit]
            final_score = int(final_score_raw.replace('_', '0'))

        # is_valid_dataをセットする
        # スコアが0,または動画の長さが1:40未満(easy/normal/hardのトリミング)のpostは無効データとして扱う
        if final_score == 0 or cnt_frame < 100 * fps:
            is_valid_data = '9'
        else:
            is_valid_data = '0'

        # stage_mode
        # correct score that exceeds 100000 and no break
        if 200000 > final_score > 100000 and stage_mode == 'n':
            final_score = final_score - 100000

        gb_posts_dict['final_score'] = final_score
        gb_posts_dict['end_score'] = end_score
        gb_posts_dict['end_score_raw'] = end_score_raw
        gb_posts_dict['total_score'] = total_score
        gb_posts_dict['total_score_raw'] = str(total_score_raw)
        gb_posts_dict['is_valid_data'] = is_valid_data
        gb_posts_dict_list_commit.append(gb_posts_dict)

        logging.info('after:')
        logging.info(gb_posts_dict)
        logging.info('analyzeMvFile/処理完了：' + str(i))

        # 検証用
        cv2.imwrite(os.path.join('./verify', '%s_%s.png' % (gb_posts_dict['id'], final_score)), proc_imgs[0])

        if i % 100 == 0:
            update_record_with_user_id(gb_posts_dict_list_commit)
            gb_posts_dict_list_commit = []

    # ramain here
    update_record_with_user_id(gb_posts_dict_list_commit)
    es_cnn.close_sess()
    ts_cnn.close_sess()

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
        logging.error(reason)

    return True


if __name__ == '__main__':
    main()
