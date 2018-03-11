# coding: utf-8
"""
lobi.play
ごまおつギルバト動画スクレイピング
"""

import os
import errno
import logging
import datetime

from sqlalchemy.orm import sessionmaker
from sqlalchemy import or_, and_

from gmot.data.DbAccessor import DbAccessor

MOVIE_DIR = '../movie'


def main():
    # prepare db
    db = DBAccessor()
    db.prepare_connect()

    series_list = []
    logging.basicConfig(
        filename='../log/{0}_{1}.log'
        .format(os.path.basename(__file__), datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
        level=logging.DEBUG, format='%(asctime)s %(message)s')

    movie_id_list = get_target_data(db)
    logging.info('delete list:')
    logging.info(movie_id_list)
    delete_movie(movie_id_list)


def get_target_data(db):
    # 既存データ取得
    Session = sessionmaker(bind=db.engine)
    session = Session()
    gb_posts_result = (session.query(
        DbAccessor.GBPost.id,
        DbAccessor.GBPost.stage_mode,
        DbAccessor.GBPost.final_score
        )
        .filter(
            or_(
                and_(DbAccessor.GBPost.final_score <= '65000',
                     DbAccessor.GBPost.stage_mode == 'n'),
                and_(DbAccessor.GBPost.final_score <= '125000',
                     DbAccessor.GBPost.stage_mode == 'b'),
            )
        )
        # .limit(10)
        # .offset(10)
        # .all()
    )

    session.flush()
    session.commit()

    movie_id_list = []
    for gb_post in gb_posts_result:
        movie_id_list.append(gb_post.id)

    # gb_posts_dict_list = []
    # for gb_post in gb_posts_result:
    #     gb_post_dict = gb_post._asdict()
    #     gb_posts_dict_list.append(gb_post_dict)
        
    return movie_id_list


def delete_movie(movie_id_list):

    for movie_id in movie_id_list:
        try:
            file_path = os.path.join(MOVIE_DIR, movie_id + '.mp4')
            os.remove(file_path)
        except OSError as e:                # this would be "except OSError, e:" before Python 2.6
            logging.warning('{} is not found'.format(movie_id))
            if e.errno != errno.ENOENT:     # errno.ENOENT = no such file or directory
                raise                       # re-raise exception if a different error occurred


def update_record_with_user_id(gb_posts_dict_list, db):

    # 既存データ取得
    Session = sessionmaker(bind=db.engine)
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
