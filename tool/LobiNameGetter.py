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

from gmot.data.DataModel import PostList, PostDetail
import gmot.data.DbAccessor as DbAccessor

def main():
    gb_posts_dict_list = getTargetData()
    gb_posts_dict_list = getLobiNameList(gb_posts_dict_list)
    updateRecordWithLobiName(gb_posts_dict_list)

def getTargetData():
    # 既存データ取得
    Session = sessionmaker(bind=DbAccessor.engine)
    session = Session()
    gb_posts_result = ( session.query(DbAccessor.GBPost.id, DbAccessor.GBPost.lobi_name)
                        .limit(8000)
                        # .all()
                        )
    session.flush()
    session.commit()

    gb_posts_dict_list = []
    for gb_post in gb_posts_result:
        gb_post_dict = gb_post._asdict()
        if gb_post_dict.get('lobi_name') == None:
            gb_posts_dict_list.append(gb_post_dict)
    return gb_posts_dict_list

def getLobiNameList(gb_posts_dict_list):
    # リクエストURL: https://play.lobi.co/video/(id)
    
    print('getLobiNameList/対象件数：' + str(len(gb_posts_dict_list)))

    remove_count = 0   
    for i, gbPostsDict in enumerate(gb_posts_dict_list):
        
        # 1秒スリープ
        time.sleep(3)

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
            lobi_name = root.cssselect('.entry-author__name--lobi')[0].text_content()
        except IndexError as e:
            print('%s;getLobiNameList:多分動画が削除されてるか、lobi紐付けがされていないので処理対象外:%s:%s'
                     % (e, i, post_url))
            remove_count += 1
            continue

        gb_posts_dict_list[i]['lobi_name'] = lobi_name.replace('Lobi:', '')

        print('getLobiNameList/処理完了：%s' % i)

    return gb_posts_dict_list

    print('getUserLobiName/全件処理完了/処理対象外件数：%s' + remove_count)


def updateRecordWithLobiName(gbPosts_dictList):

    # 既存データ取得
    Session = sessionmaker(bind=DbAccessor.engine)
    session = Session()

    # Mapping生成
    # [{
    #     'id': 1, # This is pk?
    #     'userId': 'jack@yahoo.com'
    # }, ...]
    gb_post_mappings = gbPosts_dictList

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

    return None


if __name__ == '__main__':
    main()
