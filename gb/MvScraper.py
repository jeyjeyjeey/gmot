# coding: utf-8
"""
lobi.play
ごまおつギルバト動画スクレイピング
"""

import sys
import os
import time
import datetime
import iso8601
import numpy as np

import lxml.html
import requests
import cv2
from sqlalchemy.orm import sessionmaker

sys.path.append("/Users/jeey/Dev/python/")
from gmot.data.DataModel import PostList, PostDetailGuild
import gmot.data.DbAccessor as DbAccessor
from gmot.ml.KNeighborsClassifierScikitLearn import knnTrain
from gmot.gb.MvAnalyzer import extractCapEndScore, extractCapTotalScore, extractCapMode,\
                             ajustCapture, ocrEndScore, ocrTotalScore, discernMode,\
                             KNN_IDENTIFIER_END_SCORE, KNN_IDENTIFIER_TOTAL_SCORE, KNN_IDENTIFIER_MODE

# 闘技場の識別子リスト
# 混合火と混合闇は同一識別子なので注意(曜日で判別)
META_IDS_DIC = { '火有利1':'95809', '水有利1':'94192', '風有利1':'95126', '混合火1':'97548',\
             '闇有利1':'67204', '光有利1':'101765', '混合闇1':'97548',\
             '水有利2':'240911', '風有利2':'237788',\
             '闇有利2':'240744', '光有利2':'240371'}

# パス
MOVIE_DIR = '../movie'
CAPTURE_DIR = '../capture'
TRAIN_DATA_DIR_END_SCORE = '../traindata/end_score'
TRAIN_DATA_DIR_TOTAL_SCORE = '../traindata/total_score'
TRAIN_DATA_DIR_MODE = '../traindata/mode'

# オプション
GUESS_MV_URL = False # Trueの場合、lobi_nameは取得できない
CHECK_POST_DATA_EXIST = True
DATA_INSERT = True
DEBUG = False

def main(): # 直接実行可能

    # 引数処理
    args = sys.argv
    post_list = PostList()
    if len(args) == 1:
        post_list.meta_ids_name = '闇有利1'
        post_list.rows = 1
    elif len(args) == 3:
        post_list.meta_ids_name = args[1]
        post_list.rows = int(args[2])
    else:
        print('arguments error/Usage:meta_ids_name rows')
        exit(1)

    if post_list.meta_ids_name not in META_IDS_DIC:
        print('arguments error/meta_ids_name is invalid:%s' % META_IDS_DIC.keys())
        exit(1)
    if not (0 < post_list.rows and post_list.rows <= 1000):
        print('arguments error/rows is invalid:range 1-1000')
        exit(1)
    
    post_list = getPostsList(post_list)
    post_detail_list = getPostsDetail(post_list)
    post_detail_list = editPostData(post_list, post_detail_list)
    post_detail_list = getMvUrl(post_detail_list)
    post_detail_list = downloadMvFile(post_detail_list)

    # end_score[0-9]識別用のknnオブジェクトを生成
    zero_to_ten = range(0, 10)
    detect_chrs = np.append(zero_to_ten, '_')
    knnTrain(detect_chrs, TRAIN_DATA_DIR_END_SCORE, KNN_IDENTIFIER_END_SCORE, 3)
    # total_score[0-9]識別用のknnオブジェクトを生成
    knnTrain(detect_chrs, TRAIN_DATA_DIR_TOTAL_SCORE, KNN_IDENTIFIER_TOTAL_SCORE, 3)
    # break,nobreak識別用のknnオブジェクトを生成
    detect_chrs = ['b', 'n']
    knnTrain(detect_chrs, TRAIN_DATA_DIR_MODE, KNN_IDENTIFIER_MODE, 3)

    post_detail_list = analyzeMvFile(post_detail_list)
    post_detail_list = dataCleanse(post_detail_list)
    if DATA_INSERT: insertPosts(post_detail_list)

    exit(0)

def getPostsList(post_list):
    # リクエストURL https://play.lobi.co/videos?
    # 以下パラメータ
    # 任意 rows 一ページに何件表示するか
    # 任意 meta_ids タグ
    # 任意 app アプリのidentifer ごまおつ:a74e5f31444c8498df30de2cd8d31a0d03db4f55 
    # 任意 page XXXX 何ページ目か

    # post一覧取得URL生成
    post_list_url = ( 'https://play.lobi.co/videos?meta_ids=%s&rows=%s'
                    % (META_IDS_DIC[post_list.meta_ids_name], post_list.rows) )

    # クエリ投げる
    try:
        target_html = requests.get(post_list_url).text
    except requests.exceptions.RequestException as e:
        print(str(e) + ':getPostsList/リトライはしないよ')
        exit(1)

    # fromstringはList<HtmlElement>を返す
    root = lxml.html.fromstring(target_html)

    # 投稿リストを取得
    post_list.posts = root.cssselect('.video-item')

    return post_list

def getPostsDetail(post_list):
    # リクエストURL https://play.lobi.co/video/(id)

    _post_detail_list = []
    remove_count = 0
    for post in post_list.posts:

        try:
            title_text_link = post.cssselect('.video-item__title-text')
            title = title_text_link[0].get('title')
            post_url = title_text_link[0].get('href')
            author = post.cssselect('.video-item__author__link')[0].text_content()
            user_page_link = post.cssselect('.video-item__author__link')[0].get('href')   
            post_datetime = post.cssselect('.video-item__created-date')[0].get('datetime')
            duration = post.cssselect('.video-item__duration')[0].text_content()
        except IndexError as e:
            print(str(e) + ':多分動画が圧縮されてるので、こいつは処理対象外:' + post_url)
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

        _post_detail_list.append(post_detail)

    print('getPostsDetail Finished/removeCount:' + str(remove_count))

    return _post_detail_list

def editPostData(postList, post_detail_list):

    # 1.データ存在チェック・削除
    # テーブルにすでに同一idのレコードが存在した場合、解析済みとして以降の操作対象外とする
    if CHECK_POST_DATA_EXIST:

        # 既存データ取得
        Session = sessionmaker(bind=DbAccessor.engine)
        session = Session()
        gb_posts_result = ( session.query(DbAccessor.GBPost.id)
                         .filter(DbAccessor.GBPost.meta_ids_name == postList.meta_ids_name).all() )
        session.flush()
        session.commit()

        remove_count = 0
        if len(gb_posts_result) != 0:
            gb_posts_id_list = [idTupled[0] for idTupled in gb_posts_result]

            remove_count = 0
            _post_detail_list = []
            for post_detail in post_detail_list:
                if post_detail.id in gb_posts_id_list:
                    # 追加しないことによって、取り除いてしまう
                    remove_count += 1
                else:
                    _post_detail_list.append(post_detail)

            post_detail_list = _post_detail_list
        print('editPostData:CHECK_POST_DATA_EXIST Finished/removeCount:' + str(remove_count))

    # 2.タグ整合性検証
    # タグに対して不適切な曜日に投稿されたpostを除外する
    # 1)混合火と混合闇は同一タグであるため、指定したmeta_ids_nameに適合しない曜日の場合、除外する
    # 2)新ステージ(2)は初期の頃(1)と同一タグだったため、指定したmeta_ids_nameが(1)であるのに適合しない曜日の場合、除外する
    metaIdsList = ['火有利1', '水有利1', '風有利1', '混合火1', '闇有利1', '光有利1', '混合闇1', '水有利2', '風有利2', '闇有利2', '光有利2']
    old_meta_ids_holiday_stage2 = ['水有利1', '風有利1']
    old_meta_ids_workday_stage2 = ['闇有利1', '光有利1']

    remove_count = 0
    _post_detail_list = []
    for post_detail in post_detail_list:
        post_weekday = post_detail.post_datetime.weekday()

        workday = [0, 1, 2, 3, 4]
        holiday = [5, 6]

        # 混合火1なのに休日
        if post_detail.meta_ids_name == '混合火1' and post_weekday in holiday:
            remove_count += 1 # 追加しないことによって、取り除いてしまう
        # 混合闇1なのに平日
        elif post_detail.meta_ids_name == '混合闇1' and post_weekday in workday:
            remove_count += 1 # 追加しないことによって、取り除いてしまう
        # 平日旧ステージなのに休日
        elif post_detail.meta_ids_name in old_meta_ids_holiday_stage2 and post_weekday in holiday:
            remove_count += 1 # 追加しないことによって、取り除いてしまう
        # 休日旧ステージなのに平日
        elif post_detail.meta_ids_name in old_meta_ids_workday_stage2 and post_weekday in workday:
            remove_count += 1 # 追加しないことによって、取り除いてしまう
        else:
            _post_detail_list.append(post_detail)

    post_detail_list = _post_detail_list
    print('editPostData:VALID_POST_META_IDS_NAME Finished/remove_count:' + str(remove_count))
    print('editPostData/remainCount:' + str(len(post_detail_list)))

    return post_detail_list

def getMvUrl(post_detail_list):
    # リクエストURL https://d29yz6f144inkz.cloudfront.net/(idの1文字目)/(idの2文字目)/(idの3文字目)/(idの4文字目)/(id)/video/mp4_hq.mp4

    if GUESS_MV_URL:
        for i, post_detail in enumerate(post_detail_list):
            
            id_dir = list(post_detail.id)[:4]
            mv_url = ( 'https://d29yz6f144inkz.cloudfront.net/%s/%s/%s/%s/%s/video/mp4_hq.mp4'
                    % (id_dir[0], id_dir[1], id_dir[2], id_dir[3], post_detail.id) )
            
            post_detail_list[i].mv_url = mv_url
            post_detail_list[i].lobi_name = ''

        return post_detail_list

    else:
        for i, post_detail in enumerate(post_detail_list):

            # 一秒スリープ
            time.sleep(1)
            
            if DEBUG:
                print('getMvUrl:GUESS_MV_URL_FALSE/start：%s:%s' % (i, post_detail.id))

            try:
                target_html = requests.get(post_detail.post_url).text
            except requests.exceptions.RequestException as e:
                print(str(e) + ':getPostsList/リトライはしないよ')
                exit(1)

            root = lxml.html.fromstring(target_html)
            # cssselectはHtmlElementオブジェクトのメソッド
            # List<HtmlElement>が返ってくる

            # text_content()メソッドはそのタグ以下にあるすべてのテキストを取得する
            # get()メソッドはそのタグの引数指定した属性の値を取得する
            
            
            try:
                video_elemnt = root.cssselect('source[type="video/mp4"]')
                mv_url = video_elemnt[0].get('src')
                lobi_name = root.cssselect('.entry-author__name--lobi')[0].text_content()
            except IndexError as e:
                id_dir = list(post_detail.id)[:4]
                mv_url = ( 'https://d29yz6f144inkz.cloudfront.net/%s/%s/%s/%s/%s/video/mp4_hq.mp4'
                    % (id_dir[0], id_dir[1], id_dir[2], id_dir[3], post_detail.id) )
                lobi_name = ''
                print('%s:getMvUrl:GUESS_MV_URL_FALSE/Run same with GUESS_MODE because this movie is deleted or LobiUser is not linked:%s:%s'
                        % (e, i, post_detail.post_url))

            post_detail_list[i].mv_url = mv_url
            post_detail_list[i].lobi_name = lobi_name.replace('Lobi:', '')

        return post_detail_list

def downloadMvFile(post_detail_list):

    for i, post_detail in enumerate(post_detail_list):

        #ファイル存在確認　もうあったら落とさない
        if os.path.exists(post_detail.mv_path):
            if DEBUG:
                print('downloadMvFile/動画がもうあるからDLしないよ: %i' % i)
            post_detail_list[i].mv_exist = True
        else:
            #ダウンロード
            r = requests.get(post_detail.mv_url, stream=True)
            #保存
            if r.status_code == 200:
                with open(post_detail.mv_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                            f.flush()
            if DEBUG:
                print('downloadMvFile/動画DL終わったよ: %i' % i)

    return post_detail_list

def analyzeMvFile(post_detail_list):

    remove_post_details = []
    for i, post_detail in enumerate(post_detail_list):
        
        mv = cv2.VideoCapture(post_detail.mv_path)

        # 1.end_score
        proc_imgs = extractCapEndScore(mv)
        if proc_imgs == None or len(proc_imgs) == 0:
            print('analyzeMvFile:end_score/InvalidMovie(This movie file has no frame): %s'
                    % post_detail.mv_name)
            remove_post_details.append(post_detail)
            continue
        proc_imgs = ajustCapture(proc_imgs)
        end_score_raw, end_score = ocrEndScore(proc_imgs)
        post_detail_list[i].end_score_raw = end_score_raw
        post_detail_list[i].end_score = end_score

        # 2.total_score
        proc_imgs = extractCapTotalScore(mv)
        if proc_imgs == None or len(proc_imgs) == 0:
            print('analyzeMvFile:total_score/InvalidMovie(This movie file has no frame): %s'
                    % post_detail.mv_name)
            remove_post_details.append(post_detail)
            continue
        proc_imgs = ajustCapture(proc_imgs)
        total_score, total_score_raw, total_score_count = ocrTotalScore(proc_imgs)
        post_detail_list[i].total_score = total_score
        post_detail_list[i].total_score_raw = total_score_raw
        post_detail_list[i].total_score_count = total_score_count

        # 3.break/nobreak
        proc_imgs = extractCapMode(mv)
        if proc_imgs == None or len(proc_imgs) == 0:
            print('analyzeCapture:mode/InvalidMovie(This movie file has no frame): %s'
                    % post_detail.mv_name)
            remove_post_details.append(post_detail)
            continue
        proc_imgs = ajustCapture(proc_imgs)
        stage_mode = discernMode(proc_imgs)
        post_detail_list[i].stage_mode = stage_mode

    # 無効な動画データのpost_detailをリストから除去する
    if len(remove_post_details) > 0:
        for remove_post_detail in remove_post_details:
            print('analyzeCapture/Invalid Movie Count：%s' % remove_post_detail.id)
            post_detail_list.remove(remove_post_detail)

    return post_detail_list

def dataCleanse(post_detail_list):

    for i, post in enumerate(post_detail_list):
        # final_scoreをセットする
        # total_scoreが有効である場合、total_scoreを優先する
        if post.total_score != 0 and post.total_score_count > 1: 
            post_detail_list[i].final_score = post.total_score
        else:
            post_detail_list[i].final_score = post.end_score
        # end_scoreに認識できない文字が2文字以上あるか400000以上である時、
        # total_scoreも有効でないpostは無効データとして扱う
        is_valid_data = '0'
        if ((post.end_score_raw.count('_') > 1
            or post.end_score > 400000)
            and post.total_score == 0):
                post_detail_list[i].is_valid_data = '9'
                post_detail_list[i].final_score = post.total_score
        # 動画の長さが1:40未満のpostは無効データとして扱う(easy/normalは対象外)
        if (int(post.duration.replace(':', '')) < 140):
            post_detail_list[i].is_valid_data = '9'
        # スコアが100000を超えるpostはbreakと判断する
        if post_detail_list[i].final_score > 100000:
            post_detail_list[i].stage_mode = 'b'

        if DEBUG:
            [print(key + ': ' + str(value)) for key, value in post_detail_list[i].__dict__.items()]

    return post_detail_list

def insertPosts(post_detail_list):
    Session = sessionmaker(bind=DbAccessor.engine)
    session = Session()
    now = datetime.datetime.now()

    posts = []
    for post in post_detail_list:

        gbPost = DbAccessor.GBPost(
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
            post.media,                 # media        
            '0',                        # is_final_score_editted
            post.is_valid_data          # is_valid_data  
        ) 
        posts.append(gbPost)

    session.bulk_save_objects(posts)
    session.flush()
    try:
        session.commit()
    except IntegrityError as e:
        reason = str(e)
        logger.warning(reason)

        if "Duplicate entry" in reason:
            logger.info('the inserting row already in table')
            Session.rollback()

        else:
            logger.info(reason)
            Session.rollback()
            raise e

if __name__ == '__main__':
    main()
