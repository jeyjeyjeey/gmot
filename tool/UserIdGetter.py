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

import lxml.html
import requests
import cv2
import syslog
from sqlalchemy.orm import sessionmaker

import gmot.data.DbAccessor as DbAccessor


def main():
    gb_posts_dict_list = get_target_data()
    gb_posts_dict_list = get_user_id_list(gb_posts_dict_list)
    update_record_with_user_id(gb_posts_dict_list)


def get_target_data():
    # 既存データ取得
    Session = sessionmaker(bind=DbAccessor.engine)
    session = Session()
    gb_posts_result = (session.query(DbAccessor.GBPost.id, DbAccessor.GBPost.user_id)
                       .limit(7000)
                       # .all()
                       )
    session.flush()
    session.commit()

    gb_posts_dict_list = []
    for gb_post in gb_posts_result:
        gb_post_dict = gb_post._asdict()
        if gb_post_dict.get('user_id') is None:
            gb_posts_dict_list.append(gb_post_dict)
    return gb_posts_dict_list


def get_user_id_list(gb_posts_dict_list):
    # リクエストURL: https://play.lobi.co/video/(id)
    
    print('getUserIdList/対象件数：' + str(len(gb_posts_dict_list)))

    remove_count = 0   
    for i, gbPostsDict in enumerate(gb_posts_dict_list):
        
        post_url = 'https://play.lobi.co/video/' + str(gbPostsDict.get('id'))
        try:
            target_html = requests.get(post_url).text
        except requests.exceptions.RequestException as e:
            print(str(e) + ':getUserIdList/リトライせずに次のポストを見に行くよ')
            continue

        root = lxml.html.fromstring(target_html)
        # cssselectはHtmlElementオブジェクトのメソッド
        # List<HtmlElement>が返ってくる
        try:
            user_id = root.cssselect('.entry-author__videos--link')[0].get('href').split('/')[-1]
        except IndexError as e:
            print(str(e) + ':多分動画が削除されてるので、こいつは処理対象外:' + post_url)
            continue

        # 5秒スリープ
        time.sleep(1)

        gb_posts_dict_list[i]['user_id'] = user_id

        print('getUserIdList/処理完了：' + str(i))

    print('getUserIdList/全県処理完了/処理対象外件数：' + str(remove_count))

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

    return None


if __name__ == '__main__':
    main()
