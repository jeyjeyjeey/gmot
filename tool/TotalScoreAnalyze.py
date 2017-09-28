# coding: utf-8
"""
lobi.play
ごまおつギルバト動画スクレイピング
"""

import os
import time
import datetime
import iso8601
import math
import numpy as np

import cv2
import syslog
from sqlalchemy.orm import sessionmaker

from gmot.data.DataModel import PostList, PostDetail
import gmot.data.DbAccessor as DbAccessor
from gmot.ml.KNeighborsClassifierScikitLearn import knnTrain, knnClassify
from gmot.gb.MvAnalyzer import extractCapTotalScore, ajustCapture, ocrTotalScore,\
                             KNN_IDENTIFIER_TOTAL_SCORE


MOVIE_DIR = 'movie'
TRAIN_DATA_DIR_TOTAL_SCORE = 'traindata/total_score'

def main():
    gb_posts_dict_list = getTargetData()

    # total_score[0-9]識別用のknnオブジェクトを生成
    zero_to_ten = range(0, 10)
    detect_chrs = np.append(zero_to_ten, '_')
    knnTrain(detect_chrs, TRAIN_DATA_DIR_TOTAL_SCORE, KNN_IDENTIFIER_TOTAL_SCORE, 3)

    gb_posts_dict_list = analyzeMvFile(gb_posts_dict_list)
    updateRecordWithUserId(gb_posts_dict_list)

def getTargetData():
    # 既存データ取得
    Session = sessionmaker(bind=DbAccessor.engine)
    session = Session()
    gb_posts_result = ( session.query(  DbAccessor.GBPost.id,
                                        DbAccessor.GBPost.is_valid_data,
                                        DbAccessor.GBPost.final_score,
                                        DbAccessor.GBPost.total_score,
                                        DbAccessor.GBPost.end_score,
                                        DbAccessor.GBPost.end_score_raw
                                    )
                        # .limit(7500)
                        .all()
                        )
    session.flush()
    session.commit()

    gb_posts_dict_list = []
    for gb_post in gb_posts_result:
        gb_post_dict = gb_post._asdict()
        if gb_post_dict.get('total_score') == None:
            gb_posts_dict_list.append(gb_post_dict)
        
    return gb_posts_dict_list

def analyzeMvFile(gb_posts_dict_list):
    
    print('analyzeMvFile/対象件数：' + str(len(gb_posts_dict_list)))

    remove_count = 0   
    for i, gb_posts_dict in enumerate(gb_posts_dict_list):

        print('before:')
        print(gb_posts_dict)

        mv_name = gb_posts_dict['id'] + '.mp4'
        mv = cv2.VideoCapture(os.path.join(MOVIE_DIR, mv_name))
        fps = mv.get(cv2.CAP_PROP_FPS)
        cnt_frame = mv.get(cv2.CAP_PROP_FRAME_COUNT)
        end_score = gb_posts_dict['end_score']
        end_score_raw = gb_posts_dict['end_score_raw']
        
        # 2.total_score
        proc_imgs = extractCapTotalScore(mv)
        if proc_imgs == None or len(proc_imgs) == 0:
            print('analyzeMvFile:total_score/InvalidMovie(This movie file has no frame): %s'
                    % mv_name)
            continue
        proc_imgs = ajustCapture(proc_imgs)
        total_score, total_score_raw, total_score_count = ocrTotalScore(proc_imgs)

        #dataCleanse
        # final_scoreをセットする
        # total_scoreが有効である場合、total_scoreを優先する
        if total_score != 0 and total_score_count > 1: 
            final_score = total_score
        else:
            final_score = end_score
        # is_valid_dataをセットする
        # 動画の長さが1:40未満のpostは無効データとして扱う(easy/normalは対象外)
        is_valid_data = '0'
        if ((end_score_raw.count('_') > 1
            or end_score > 400000)
            and total_score == 0):
                is_valid_data = '9'
                final_score = total_score
        if (cnt_frame < 100 * fps):
            is_valid_data = '9'

        gb_posts_dict_list[i]['final_score'] = final_score
        gb_posts_dict_list[i]['total_score'] = total_score
        gb_posts_dict_list[i]['total_score_raw'] = total_score_raw
        gb_posts_dict_list[i]['is_valid_data'] = is_valid_data

        print('after:')
        print(gb_posts_dict)
        print('analyzeMvFile/処理完了：' + str(i))

    return gb_posts_dict_list


def updateRecordWithUserId(gb_posts_dictList):

    # 既存データ取得
    Session = sessionmaker(bind=DbAccessor.engine)
    session = Session()

    # Mapping生成
    # [{
    #     'id': 1,
    #     'userId': 'hoge@hoge.com'
    # }, ...]
    gb_post_mappings = gb_posts_dictList

    session.bulk_update_mappings(DbAccessor.GBPost, gb_post_mappings)
    session.flush()

    try:
        session.commit()
    except Exception as e:
        reason = str(e)
        syslog.warning(reason)

        if "Duplicate entry" in reason:
            syslog.info('the inserting row already in table')
            Session.rollback()

        else:
            syslog.info(reason)
            Session.rollback()
            raise e

    return True


if __name__ == '__main__':
    main()
