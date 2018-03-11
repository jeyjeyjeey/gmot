# coding: utf-8
"""
Scraping gmot guild battle movie
for lobi.play
"""

import sys
import os
import logging
import time
import pytz
import datetime
import iso8601
import numpy as np

import lxml.html
import requests
import cv2
from sqlalchemy.orm import sessionmaker

from gmot.data.DataModel import PostList, PostDetailGuild
from gmot.data.DbAccessor import DBAccessor, GBPost
from mlsp.ml.KNeighborsClassifierScikitLearn import knn_train, knn_teardown_all
from mlsp.ml.CNNClassifierDigit import CNNClassifierDigit
from gmot.gb.MvAnalyzer import extract_cap_end_score, extract_cap_total_score, extract_cap_mode,\
                             clip_caputure, ajust_capture, ocr_end_score_cnn, ocr_total_score_cnn, discern_mode,\
                             KNN_IDENTIFIER_MODE

logger = logging.getLogger(__name__)

# constant
# attention:混合火 and 混合闇 is same identifier(Determined by day of the week
META_IDS_DIC = {'火有利1': '95809', '水有利1': '94192', '風有利1': '95126', '混合火1': '97548',
                        '闇有利1': '67204', '光有利1': '101765', '混合闇1': '97548',
                        '水有利2': '240911', '風有利2': '237788',
                        '闇有利2': '240744', '光有利2': '240371', '火有利2': '270488'}
DATETIME_FIRE2_ADD = datetime.datetime(2017, 10, 31, tzinfo=pytz.timezone('Asia/Tokyo'))

# optional settings
GUESS_MV_URL = False  # when true, cannot get lobi_name
CHECK_POST_DATA_EXIST = True
DATA_INSERT = True

# default dir
MOVIE_DIR = '../movie'
CAPTURE_DIR = '../capture'


def main():
    # directory path
    TRAIN_DATA_DIR_MODE = '../traindata/mode'
    TRAIN_DATA_DIR_END_SCORE_CNN = '../traindata/end_score_cnn'
    CKPT_DATA_DIR_END_SCORE_CNN = '../ckpt/end_score'
    TRAIN_DATA_DIR_TOTAL_SCORE_CNN = '../traindata/total_score_cnn'
    CKPT_DATA_DIR_TOTAL_SCORE_CNN = '../ckpt/total_score'

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S'
                        )

    # Args preparing
    args = sys.argv
    post_list = PostList()
    if len(args) == 1:
        post_list.meta_ids_name = '水有利1'
        post_list.rows = 100
    elif len(args) == 3:
        post_list.meta_ids_name = args[1]
        post_list.rows = int(args[2])
    else:
        logger.error('arguments error/Usage:meta_ids_name rows')
        exit(1)
    if post_list.meta_ids_name not in META_IDS_DIC:
        logger.error('arguments error/meta_ids_name is invalid:%s' % META_IDS_DIC.keys())
        exit(1)
    if not (0 < post_list.rows <= 1000):
        logger.error('arguments error/rows is invalid:range 1-1000')
        exit(1)

    # prepare db
    db = DBAccessor()
    db.prepare_connect()

    # Scraping
    post_list = get_posts_list(post_list)
    if post_list is None:
        exit(1)
    post_detail_list = get_posts_detail(post_list)
    post_detail_list = edit_post_data(post_list, post_detail_list, db)
    post_detail_list = get_mv_url(post_detail_list)
    if post_detail_list is None:
        exit(1)
    post_detail_list = download_mv_file(post_detail_list)
    if post_detail_list is None:
        exit(1)

    # When there is no target, end here
    if len(post_detail_list) == 0:
        exit(0)

    # KNNClassfier preparing
    # # end_score[0-9]
    # zero_to_ten = range(0, 10)
    # detect_chrs = np.append(zero_to_ten, '_')
    # knn_train(detect_chrs, TRAIN_DATA_DIR_END_SCORE, KNN_IDENTIFIER_END_SCORE, 3)
    # # total_score[0-9]
    # knn_train(detect_chrs, TRAIN_DATA_DIR_TOTAL_SCORE, KNN_IDENTIFIER_TOTAL_SCORE, 3)
    # break,nobreak
    detect_chrs = ['b', 'n']
    knn_train(detect_chrs, TRAIN_DATA_DIR_MODE, KNN_IDENTIFIER_MODE, 3)

    # Movie analyze
    # CNN prepare
    es_cnn = CNNClassifierDigit()
    es_cnn.ckpt_dir = CKPT_DATA_DIR_END_SCORE_CNN
    es_cnn.prepare_classify()
    ts_cnn = CNNClassifierDigit()
    ts_cnn.ckpt_dir = CKPT_DATA_DIR_TOTAL_SCORE_CNN
    ts_cnn.prepare_classify()

    post_detail_list = analyze_mv_file(post_detail_list, es_cnn, ts_cnn)
    knn_teardown_all()

    # Data persistence
    post_detail_list = data_cleanse(post_detail_list)
    if DATA_INSERT:
        insert_posts(post_detail_list, db)

    exit(0)


def get_posts_list(post_list):
    # requestURL lobi
    # https://play.lobi.co/videos?
    # -parameters
    # either_required meta_ids :tags
    # either_required app :app_identifier gmot:a74e5f31444c8498df30de2cd8d31a0d03db4f55
    # optional rows :display per page
    # optional page :page number

    post_list_url = ('https://play.lobi.co/videos?meta_ids=%s&rows=%s'
                     % (META_IDS_DIC[post_list.meta_ids_name], post_list.rows))

    try:
        target_html = requests.get(post_list_url).text
    except requests.exceptions.RequestException as e:
        logger.error(str(e) + ':getPostsList/no retry')
        return None

    # Method fromstring return List<HtmlElement>
    root = lxml.html.fromstring(target_html)
    post_list.posts = root.cssselect('.video-item')

    return post_list


def get_posts_detail(post_list):
    # requestURL lobi
    # https://play.lobi.co/video/(id)

    _post_detail_list = []
    remove_count = 0
    for post in post_list.posts:

        post_url = 'url取得失敗'
        try:
            title_text_link = post.cssselect('.video-item__title-text')
            title = title_text_link[0].get('title')
            post_url = title_text_link[0].get('href')
            author = post.cssselect('.video-item__author__link')[0].text_content()
            user_page_link = post.cssselect('.video-item__author__link')[0].get('href')   
            post_datetime = post.cssselect('.video-item__created-date')[0].get('datetime')
            duration = post.cssselect('.video-item__duration')[0].text_content()
        except IndexError as e:
            logger.info(str(e) + ':多分動画が圧縮されてるので、こいつは処理対象外:' + post_url)
            remove_count += 1
            continue

        post_detail = PostDetailGuild()
        post_detail.title = title
        post_detail.post_url = post_url
        post_detail.author = author
        post_detail.user_id = user_page_link.split('/')[-1]
        post_detail.post_datetime = iso8601.parse_date(post_datetime)
        post_detail.duration = duration
        post_detail.id = post_detail.post_url.split('/')[-1]
        post_detail.meta_ids_name = post_list.meta_ids_name
        post_detail.mv_name = post_detail.id + '.mp4'
        post_detail.mv_path = os.path.join(MOVIE_DIR, post_detail.mv_name)
        post_detail.media = 'L'
        post_detail.ring = 'C'  # circle

        _post_detail_list.append(post_detail)

    logger.info('getPostsDetail Finished/removeCount:' + str(remove_count))

    return _post_detail_list


def edit_post_data(post_list, post_detail_list, db):

    # 1.Data Existence Check
    # テーブルにすでに同一idのレコードが存在した場合、解析済みとして以降の操作対象外とする
    if CHECK_POST_DATA_EXIST:

        session_maker = sessionmaker(bind=db.engine)
        session = session_maker()
        gb_posts_result = (session.query(GBPost.id)
                           # .filter(DbAccessor.GBPost.meta_ids_name == post_list.meta_ids_name)
                           .all())
        session.flush()
        session.commit()

        remove_count = 0
        if len(gb_posts_result) != 0:
            gb_posts_id_list = [idTupled[0] for idTupled in gb_posts_result]

            remove_count = 0
            _post_detail_list = []
            for post_detail in post_detail_list:
                if post_detail.id in gb_posts_id_list:
                    remove_count += 1
                else:
                    # when error, remove by not adding
                    _post_detail_list.append(post_detail)

            post_detail_list = _post_detail_list
        logger.info('editPostData:CHECK_POST_DATA_EXIST Finished/removeCount:' + str(remove_count))

    # 2.Consistency Check of Tag
    # タグに対して不適切な曜日に投稿されたpostを除外する
    # 1)混合火と混合闇は同一タグであるため、指定したmeta_ids_nameに適合しない曜日の場合、除外する
    # 2)新ステージ(2)は初期の頃(1)と同一タグだったため、指定したmeta_ids_nameが(1)であるのに適合しない曜日の場合、除外する
    # 火有利2はどうしようねえ...
    old_meta_ids_holiday_stage2 = ['水有利1', '風有利1']
    old_meta_ids_workday_stage2 = ['闇有利1', '光有利1']

    remove_count = 0
    _post_detail_list = []
    for post_detail in post_detail_list:
        post_weekday = post_detail.post_datetime.weekday()

        workday = [0, 1, 2, 3, 4]
        holiday = [5, 6]

        # mixed_fire1 but holiday
        if post_detail.meta_ids_name == '混合火1' and post_weekday in holiday:
            remove_count += 1
        # mixed_dark1 but workday
        elif post_detail.meta_ids_name == '混合闇1' and post_weekday in workday:
            remove_count += 1
        # workday old stage but holiday
        elif post_detail.meta_ids_name in old_meta_ids_holiday_stage2 and post_weekday in holiday:
            remove_count += 1
        # holiday old stage but workday
        elif post_detail.meta_ids_name in old_meta_ids_workday_stage2 and post_weekday in workday:
            remove_count += 1
        # exclude fire2 posts whose post_datetime is before added date.
        elif post_detail.meta_ids_name == '火有利2' and post_detail.post_datetime < DATETIME_FIRE2_ADD:
            remove_count += 1
        else:
            # when error, remove by not adding
            _post_detail_list.append(post_detail)

    post_detail_list = _post_detail_list
    logger.info('editPostData:VALID_POST_META_IDS_NAME Finished/remove_count:' + str(remove_count))
    logger.info('editPostData/remainCount:' + str(len(post_detail_list)))

    return post_detail_list


def get_mv_url(post_detail_list):
    # requestURL lobi
    # https://d29yz6f144inkz.cloudfront.net/(1st of id)/(2nd)/(3rd)/(4th)/(id)/video/mp4_hq.mp4

    if GUESS_MV_URL:
        for i, post_detail in enumerate(post_detail_list):
            
            id_dir = list(post_detail.id)[:4]
            mv_url = ('https://d29yz6f144inkz.cloudfront.net/%s/%s/%s/%s/%s/video/mp4_hq.mp4'
                      % (id_dir[0], id_dir[1], id_dir[2], id_dir[3], post_detail.id))
            
            post_detail_list[i].mv_url = mv_url
            post_detail_list[i].lobi_name = ''

        return post_detail_list

    else:
        for i, post_detail in enumerate(post_detail_list):

            # 1sec sleep
            time.sleep(1)

            logger.debug('getMvUrl:GUESS_MV_URL_FALSE/start：%s:%s' % (i, post_detail.id))

            try:
                target_html = requests.get(post_detail.post_url).text
            except requests.exceptions.RequestException as e:
                logger.error(str(e) + ':getPostsList/no retry')
                return None

            root = lxml.html.fromstring(target_html)
            # cssselect is a Method of HtmlElement Object
            # It return List<HtmlElement>
            # Method "text_content" gets all texts below specified tag
            # Method "get" gets attribut value of specified tag
            
            try:
                video_elemnt = root.cssselect('source[type="video/mp4"]')
                mv_url = video_elemnt[0].get('src')
                lobi_name = root.cssselect('.entry-author__name--lobi')[0].text_content()
            except IndexError as e:
                id_dir = list(post_detail.id)[:4]
                mv_url = ('https://d29yz6f144inkz.cloudfront.net/%s/%s/%s/%s/%s/video/mp4_hq.mp4'
                          % (id_dir[0], id_dir[1], id_dir[2], id_dir[3], post_detail.id))
                lobi_name = ''
                logger.info('%s:getMvUrl:GUESS_MV_URL_FALSE/Run GUESS_MODE '
                            'because this movie is deleted or LobiUser is not linked:%s:%s'
                            % (e, i, post_detail.post_url))

            post_detail_list[i].mv_url = mv_url
            post_detail_list[i].lobi_name = lobi_name.replace('Lobi:', '')

        return post_detail_list


def download_mv_file(post_detail_list):

    for i, post_detail in enumerate(post_detail_list):

        if os.path.exists(post_detail.mv_path):
            logger.debug('downloadMvFile/for MvFile already exists, do not download: %i' % i)
            post_detail_list[i].mv_exist = True
        else:
            try:
                r = requests.get(post_detail.mv_url, stream=True)
            except requests.exceptions.RequestException as e:
                logger.error(str(e) + ':download_mv_file/no retry')
                return None
            if r.status_code == 200:
                with open(post_detail.mv_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                            f.flush()
            logger.debug('downloadMvFile/finished download mvFile: %i' % i)

    return post_detail_list


def analyze_mv_file(post_detail_list, es_cnn, ts_cnn):

    remove_post_details = []
    for i, post_detail in enumerate(post_detail_list):
        
        mv = cv2.VideoCapture(post_detail.mv_path)

        # 1.end_score
        proc_imgs = extract_cap_end_score(mv)
        if proc_imgs is None or len(proc_imgs) == 0:
            logger.warning('analyzeMvFile:end_score/InvalidMovie(This movie file has no frame): %s'
                           % post_detail.mv_name)
            remove_post_details.append(post_detail)
            continue
        proc_imgs = clip_caputure(proc_imgs)
        proc_imgs = ajust_capture(proc_imgs)
        (post_detail_list[i].end_score_raw,
         post_detail_list[i].end_score,
         post_detail_list[i].end_score_prediction) = ocr_end_score_cnn(proc_imgs, es_cnn)

        # 2.total_score
        proc_imgs = extract_cap_total_score(mv)
        if proc_imgs is None or len(proc_imgs) == 0:
            logger.warning('analyzeMvFile:total_score/InvalidMovie(This movie file has no frame): %s'
                           % post_detail.mv_name)
            remove_post_details.append(post_detail)
            continue
        proc_imgs = clip_caputure(proc_imgs)
        proc_imgs = ajust_capture(proc_imgs)
        (post_detail_list[i].total_score,
         post_detail_list[i].total_score_raw,
         post_detail_list[i].total_score_count,
         post_detail_list[i].total_score_prediction) = ocr_total_score_cnn(proc_imgs, ts_cnn)

        # 3.break/nobreak
        proc_imgs = extract_cap_mode(mv)
        if proc_imgs is None or len(proc_imgs) == 0:
            logger.warning('analyzeCapture:mode/InvalidMovie(This movie file has no frame): %s'
                           % post_detail.mv_name)
            remove_post_details.append(post_detail)
            continue
        proc_imgs = clip_caputure(proc_imgs)
        proc_imgs = ajust_capture(proc_imgs)
        post_detail_list[i].stage_mode = discern_mode(proc_imgs)

    es_cnn.close_sess()
    ts_cnn.close_sess()

    # Remove post whose movie is invalid from process post list
    if len(remove_post_details) > 0:
        for remove_post_detail in remove_post_details:
            logger.warning('analyzeCapture/Invalid Movie Count：%s' % remove_post_detail.id)
            post_detail_list.remove(remove_post_detail)

    return post_detail_list


def data_cleanse(post_detail_list):

    for i, post in enumerate(post_detail_list):
        # final_score
        # If TS/ES is unidentifiable, the other is adopted
        if post.total_score == 0:
            post_detail_list[i].final_score = post.end_score
        elif post.end_score == 0:
            post_detail_list[i].final_score = post.total_score
        # For each digit, to adopt a digit with a higher prediction
        else:
            cmpr_scr_list = np.array([post.end_score_raw, post.total_score_raw])
            cmpr_pred_list = np.array([post.end_score_prediction, post.total_score_prediction])
            if post.total_score_count <= 1:  # When source frame of total_score is few, decrease prediction rate
                cmpr_pred_list[1] = cmpr_pred_list[1] - 0.05
            max_pred_column_indices = np.argmax(cmpr_pred_list, axis=0).reshape(-1).tolist()
            final_score_raw = str()
            for which_digit, which_score in enumerate(max_pred_column_indices):
                final_score_raw += cmpr_scr_list[which_score][which_digit]
            post_detail_list[i].final_score = int(final_score_raw.replace('_', '0'))

        # is_valid_data
        # Set post invalid with score 0 or less than 1:40(maybe easy/normal/hard trimmed)
        if post_detail_list[i].final_score == 0 or int(post.duration.replace(':', '')) < 140:
                post_detail_list[i].is_valid_data = '9'
        else:
            post_detail_list[i].is_valid_data = '0'

        # stage_mode
        # Correct stage_mode that exceeds 100000 and no break
        # (Score Prediction is higher than stage mode prediction)
        if 200000 > post_detail_list[i].final_score > 100000 and post.stage_mode == 'n':
            post_detail_list[i].stage_mode = 'b'

        [logger.debug(key + ': ' + str(value)) for key, value in post_detail_list[i].__dict__.items()]

    return post_detail_list


def insert_posts(post_detail_list, db):
    session_maker = sessionmaker(bind=db.engine)
    session = session_maker()
    now = datetime.datetime.now()

    posts = []
    for post in post_detail_list:

        gb_post = GBPost(
            now,                        # created                 
            '0000-00-00 00:00:00',      # modified                
            post.id,                    # id                      
            post.post_datetime.date(),  # post_date                
            post.meta_ids_name,         # meta_ids_name           
            post.author,                # author
            post.lobi_name,             # lobi_name
            post.user_id,               # user_id                       
            post.final_score,           # final_score
            post.end_score,             # end_score
            post.end_score_raw,         # end_score_raw
            post.total_score,           # total_score
            post.total_score_raw,       # total_score_raw
            post.stage_mode,            # stage_mode
            post.post_datetime,         # post_datetime           
            post.duration,              # duration
            post.ring,                  # ring
            post.media,                 # media
            '0',                        # is_final_score_edited
            post.is_valid_data,         # is_valid_data
            None                        # stage_mode_re
        ) 
        posts.append(gb_post)

    session.bulk_save_objects(posts)
    session.flush()
    try:
        session.commit()
    except Exception as e:
        reason = str(e)

        if "Duplicate entry" in reason:
            logger.error('the inserting row already in table')
            session.rollback()

        else:
            logger.error(reason)
            session.rollback()
            raise e


if __name__ == '__main__':
    main()
