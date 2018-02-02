# coding: utf-8
"""
lobi.play
ごまおつギルバト動画スクレイピング
"""

import os
import logging
import datetime
from math import floor
import functools
import numpy as np
import pandas as pd
from PIL import Image
from MvScraper import META_IDS_DIC

import cv2
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt

import gmot.data.DbAccessor as DbAccessor
from gmot.ml.CNNClassifierDigit import CNNClassifierDigit
from gmot.gb.MvAnalyzer import ajust_capture, clip_caputure, get_valid_last_frame
from gmot.ml.KNeighborsClassifierScikitLearn import knn_train

MOVIE_DIR = '../movie'
CAP_DIR = '../capture'

X = []
Y = []
total_count = 1

IMG_OUT = False
IMG_SHOW = False

SCENE_BASE_FPS = 30
SCENE_BASE_DURATION = 120 * SCENE_BASE_FPS
SCENE_BASE_DATA = {
    "L1": {
        "fps": SCENE_BASE_FPS,
        "frame_duration": SCENE_BASE_DURATION,
        "start_frame": 20 * SCENE_BASE_FPS,
        "search_frame_count": 15 * SCENE_BASE_FPS
    },
    "L2": {
        "fps": SCENE_BASE_FPS,
        "frame_duration": SCENE_BASE_DURATION,
        "start_frame": 37 * SCENE_BASE_FPS,
        "search_frame_count": 15 * SCENE_BASE_FPS
    },
    "L3": {
        "fps": SCENE_BASE_FPS,
        "frame_duration": SCENE_BASE_DURATION,
        "start_frame": 58 * SCENE_BASE_FPS,
        "search_frame_count": 15 * SCENE_BASE_FPS
    },
    "BS": {
        "fps": SCENE_BASE_FPS,
        "frame_duration": SCENE_BASE_DURATION,
        "start_frame": 75 * SCENE_BASE_FPS,
        "search_frame_count": 15 * SCENE_BASE_FPS
    }
}


def main():
    global X, Y, total_count
    series_list = []
    logging.basicConfig(
        # filename='../log/{0}_{1}.log'
        # .format(os.path.basename(__file__), datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
        level=logging.DEBUG, format='%(asctime)s %(message)s')

    for meta_ids_name in META_IDS_DIC.keys():
        gb_posts_dict_list = get_target_data(meta_ids_name, 10)
        for gb_posts_dict in gb_posts_dict_list:
            for scene_name in ('L1', 'L2', 'L3', 'BS'):
                series_list.extend(analyze_mv_file([gb_posts_dict], scene_name))
            total_count += 1

    df = pd.DataFrame(series_list)
    df.to_csv('../log/df_{0}.csv'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))

    fig, ax = plt.subplots()
    # ax.set_xlim(0, 1)
    ax.plot(X, Y, 'ro', picker=3)
    fig.canvas.mpl_connect('pick_event', functools.partial(on_pick, series_list=series_list))
    plt.show()


def on_pick(event, series_list):
    # line = event.artist
    # xdata, ydata = line.get_data()
    ind = event.ind
    print('on_pick_object_in_line:', series_list[ind[0]])


def plot_data(pos_assumed_ratio):
    global total_count
    X.append(pos_assumed_ratio)
    Y.append(total_count)


def get_target_data(meta_ids_name, count):
    # 既存データ取得
    Session = sessionmaker(bind=DbAccessor.engine)
    session = Session()
    gb_posts_result = (session.query(DbAccessor.GBPost.id,
                                     DbAccessor.GBPost.duration,
                                     DbAccessor.GBPost.meta_ids_name,
                                     DbAccessor.GBPost.is_valid_data
                                     )
                       .filter(DbAccessor.GBPost.duration >= '02:00',
                               DbAccessor.GBPost.meta_ids_name == meta_ids_name,
                               DbAccessor.GBPost.is_valid_data == '0')
                       .limit(count)
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


def analyze_mv_file(gb_posts_dict_list, scene_name):
    logging.info('analyzeMvFile/対象件数：' + str(len(gb_posts_dict_list)))

    series_list = []
    remove_count = 0
    for i, gb_posts_dict in enumerate(gb_posts_dict_list):

        # print('before:')
        # print(gb_posts_dict)
        series = {}
        series['id'] = gb_posts_dict['id']
        series['duration'] = gb_posts_dict['duration']
        series['meta_ids_name'] = gb_posts_dict['meta_ids_name']

        mv_name = gb_posts_dict['id'] + '.mp4'
        mv = cv2.VideoCapture(os.path.join(MOVIE_DIR, mv_name))

        proc_imgs, series = extract_cap_scene(mv, scene_name, series)
        if proc_imgs is None or len(proc_imgs) == 0:
            logging.warning('analyzeMvFile:end_score/InvalidMovie(This movie file has no frame): %s'
                            % mv_name)
            continue
        proc_imgs = clip_caputure(proc_imgs)
        proc_imgs = ajust_capture(proc_imgs)

        if IMG_SHOW:
            [Image.fromarray(np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).show() for img in proc_imgs]
        if IMG_OUT:
            label = '{0}_{1}'.format(gb_posts_dict["meta_ids_name"], scene_name)
            sample_id = gb_posts_dict["id"]
            os.makedirs(os.path.join(CAP_DIR, label, sample_id), exist_ok=True)
            for num, img in enumerate(proc_imgs):
                cv2.imwrite(os.path.join(
                    CAP_DIR, label, sample_id,
                    '{0}_{1:03d}_{2}.png'.format(label, num, sample_id)
                ),
                    img)

        logging.info('after:%s', gb_posts_dict)
        logging.info('analyzeMvFile/処理完了：' + str(i))
        series_list.append(series)

    return series_list


def extract_cap_scene(mv, scene_name, series):
    SEARCH_FRAME_RATE = 0.25
    mv_fps = mv.get(cv2.CAP_PROP_FPS)

    assumed_start_frame, assumed_end_frame, mv_frame_duration, correct_variation, pos_assumed_ratio =\
        calculate_scene_time_assumed(mv, scene_name)
    frame_rate = mv_fps * correct_variation * SEARCH_FRAME_RATE

    scene_imgs = []
    for pos_frame in np.around(np.arange(assumed_start_frame, assumed_end_frame, frame_rate)):
        mv.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)

        ret, frame = mv.read()
        if ret is True:
            scene_imgs.append(frame)
        elif ret is False:
            break

    series['scene_name'] = scene_name
    series['mv_fps'] = mv_fps
    series['mv_frame_duration'] = mv_frame_duration
    series['correct_variation'] = correct_variation
    series['assumed_start_frame'] = assumed_start_frame
    series['assumed_end_frame'] = assumed_end_frame
    series['pos_assumed_ratio'] = pos_assumed_ratio
    series['frame_rate'] = frame_rate
    logging.debug('scene_name:{} / mv_frame_duration:{} / mv_fps:{} / frame_rate:{}/ correct_variation:{}'
                  .format(scene_name, mv_frame_duration, mv_fps, frame_rate, correct_variation))
    logging.debug('assumed_start_frame: {} / assumed_end_frame:{} / pos_assumed_ratio:{}'
                  .format(assumed_start_frame, assumed_end_frame, pos_assumed_ratio))

    return scene_imgs, series


def calculate_scene_time_assumed(mv, scene_name):
    mv_frame_duration = get_valid_last_frame(mv)
    scene_base_data = SCENE_BASE_DATA[scene_name]

    # Calculate how much the duration of the target movie equivalent to the length of the base.
    correct_variation = mv_frame_duration / scene_base_data["frame_duration"]
    search_frames = scene_base_data["search_frame_count"] * correct_variation
    assumed_start_frame = floor(
        (scene_base_data["start_frame"] * correct_variation) -
        (search_frames / 2)
    )
    assumed_end_frame = floor(
        assumed_start_frame + search_frames
    )

    pos_assumed_ratio = assumed_start_frame / mv_frame_duration
    plot_data(pos_assumed_ratio)

    return assumed_start_frame, assumed_end_frame, mv_frame_duration, correct_variation, pos_assumed_ratio


def estimate_scene_time():
    """
    CONVOLUTIONAL LSTM PREDICTION
    """
    pass


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
