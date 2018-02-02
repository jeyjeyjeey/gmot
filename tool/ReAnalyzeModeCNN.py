# coding: utf-8
"""
lobi.play
ごまおつギルバト動画スクレイピング
"""

import os
import logging
import datetime

import cv2
from sqlalchemy.orm import sessionmaker

import gmot.data.DbAccessor as DbAccessor
from gmot.gb.MvAnalyzer import extract_cap_mode, ajust_capture, discern_mode_cnn, clip_caputure
from gmot.ml.CNNClassifierStaticObject import CNNClassifierStaticObject


MOVIE_DIR = '../movie'
TRAIN_DATA_DIR_MODE = '../traindata/mode'


def main():
    logging.basicConfig(
        filename='../log/%s_%s.log'
        % (os.path.basename(__file__), datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
        level=logging.DEBUG, format='%(asctime)s %(message)s')

    gb_posts_dict_list = get_target_data()

    gb_posts_dict_list = analyze_mv_file(gb_posts_dict_list)
    # update_record_with_user_id(gb_posts_dict_list)


def get_target_data():
    # 既存データ取得
    Session = sessionmaker(bind=DbAccessor.engine)
    session = Session()
    gb_posts_result = (session.query(DbAccessor.GBPost.id,
                                     DbAccessor.GBPost.is_valid_data,
                                     DbAccessor.GBPost.final_score,
                                     DbAccessor.GBPost.stage_mode
                                     )
                       # .filter(DbAccessor.GBPost.id == 'ff01fd2d547d739e6b4992ce3432a6a73e74f57e')
                       # .filter(DbAccessor.GBPost.updated_at < '2017-10-17 09:00:00')
                       .filter(DbAccessor.GBPost.stage_mode_re == None)
                       .limit(1000)
                       # .all()
                       )
    session.flush()
    session.commit()

    gb_posts_dict_list = []
    for gb_post in gb_posts_result:
        gb_post_dict = gb_post._asdict()
        # if gb_post_dict.get('stage_mode_re') is None:
        gb_posts_dict_list.append(gb_post_dict)
        
    return gb_posts_dict_list


def analyze_mv_file(gb_posts_dict_list):
    
    logging.info('analyzeMvFile/対象件数：' + str(len(gb_posts_dict_list)))

    md_cnn = CNNClassifierStaticObject()
    md_cnn.weight_dir = '../weight'
    md_cnn.identifier = 'mode'
    md_cnn.classes = ['b', 'n']
    md_cnn.prepare_classify()

    gb_posts_dict_list_commit = []
    for i, gb_posts_dict in enumerate(gb_posts_dict_list):

        logging.info('before:')
        logging.info(gb_posts_dict)

        mv_name = gb_posts_dict['id'] + '.mp4'
        mv = cv2.VideoCapture(os.path.join(MOVIE_DIR, mv_name))
        final_score = gb_posts_dict['final_score']
        
        # 3.break/nobreak
        proc_imgs = extract_cap_mode(mv)
        if proc_imgs is None or len(proc_imgs) == 0:
            logging.warning('analyzeCapture:mode/InvalidMovie(This movie file has no frame): %s'
                            % mv_name)
            continue
        proc_imgs = clip_caputure(proc_imgs)
        proc_imgs = ajust_capture(proc_imgs)
        stage_mode, predictions = discern_mode_cnn(proc_imgs, md_cnn)
        logging.debug(predictions)

        # dataCleanse
        # stage_mode
        # Correct stage_mode that exceeds 100000 and no break
        if 200000 > final_score > 100000 and stage_mode == 'n':
            stage_mode = 'b'

        gb_posts_dict['stage_mode_re'] = stage_mode
        gb_posts_dict_list_commit.append(gb_posts_dict)

        logging.info('after:')
        logging.info(gb_posts_dict)
        logging.info('analyzeMvFile/処理完了：' + str(i))

        if i % 30 == 0:
            update_record_with_user_id(gb_posts_dict_list_commit)
            gb_posts_dict_list_commit = []

    # ramain here
    update_record_with_user_id(gb_posts_dict_list_commit)

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
