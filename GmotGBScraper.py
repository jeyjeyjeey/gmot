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

from GmotDataModel import PostList, PostDetail
from sqlalchemy.orm import sessionmaker
import GmotDbAccessor
from KNeighborsClassifierScikitLearn import knnTrain, knnClassify

# 闘技場の識別子リスト
# 混合火と混合闇は同一識別子なので注意(曜日で判別)
meta_ids_dic = { '火有利1':'95809', '水有利1':'94192', '風有利1':'95126', '混合火1':'97548',\
             '闇有利1':'67204', '光有利1':'101765', '混合闇1':'97548',\
             '水有利2':'240911', '風有利2':'237788',\
             '闇有利2':'240744', '光有利2':'240371'}
MOVIE_DIR = 'movie'
CAPTURE_DIR = 'capture'
TRAIN_DATA_DIR_SCORE = 'traindata/score'
TRAIN_DATA_DIR_MODE = 'traindata/mode'
KNN_IDENTIFIER_SCORE = 'score'
KNN_IDENTIFIER_MODE = 'mode'

# 実行時オプション
GUESS_MV_URL = False # Trueの場合、lobi_nameは取得できない
CHECK_POST_DATA_EXIST = True
DATA_INSERT = True

def main():
    post_list = PostList()
    post_list.meta_ids_name = '闇有利1'
    post_list.rows = '100'

    post_list = getPostsList(post_list)
    post_detail_list = getPostsDetail(post_list)
    post_detail_list = editPostData(post_list, post_detail_list)
    post_detail_list = getMvUrl(post_detail_list)
    post_detail_list = downloadMvFile(post_detail_list)
    post_detail_list = extractCapture(post_detail_list)
    post_detail_list = ajustCapture(post_detail_list)

    # [0-9]識別用のknnオブジェクトを生成
    zero_to_ten = range(0, 10)
    detect_chrs = np.append(zero_to_ten, '_')
    knnTrain(detect_chrs, TRAIN_DATA_DIR_SCORE, KNN_IDENTIFIER_SCORE, 3)
    # break,nobreak識別用のknnオブジェクトを生成
    detect_chrs = ['b', 'n']
    knnTrain(detect_chrs, TRAIN_DATA_DIR_MODE, KNN_IDENTIFIER_MODE, 3)

    post_detail_list = ocrScore(post_detail_list)
    post_detail_list = discernMode(post_detail_list)
    if DATA_INSERT: insertPosts(post_detail_list)

def getPostsList(post_list):
    # リクエストURL https://play.lobi.co/videos?
    # 以下パラメータ
    # 任意 rows 一ページに何件表示するか
    # 任意 meta_ids タグ
    # 任意 app アプリのidentifer ごまおつ:a74e5f31444c8498df30de2cd8d31a0d03db4f55 
    # 任意 page XXXX 何ページ目か

    # post一覧取得URL生成
    post_list_url = ( 'https://play.lobi.co/videos?meta_ids=%s&rows=%s'
                    % (meta_ids_dic[post_list.meta_ids_name], post_list.rows) )

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

        post_detail = PostDetail()
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

        _post_detail_list.append(post_detail)

    print('getPostsDetail Finished/removeCount:' + str(remove_count))

    return _post_detail_list

def editPostData(postList, post_detail_list):

    # 1.データ存在チェック・削除
    # テーブルにすでに同一idのレコードが存在した場合、解析済みとして以降の操作対象外とする
    if CHECK_POST_DATA_EXIST:

        # 既存データ取得
        Session = sessionmaker(bind=GmotDbAccessor.engine)
        session = Session()
        gb_posts_result = ( session.query(GmotDbAccessor.GBPost.id)
                         .filter(GmotDbAccessor.GBPost.meta_ids_name == postList.meta_ids_name).all() )
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
    print('editPostData/残り処理件数:' + str(len(post_detail_list)))

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
            
            print('getMvUrl:GUESS_MV_URL_FALSE/処理開始：%s:%s' % (i, post_detail.id))

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
                print('%s:getMvUrl:GUESS_MV_URL_FALSE/多分動画が削除されてるか、lobi紐付けがされていないためGUESSと同じ動作をする:%s:%s'
                        % (e, i, post_detail.post_url))

            post_detail_list[i].mv_url = mv_url
            post_detail_list[i].lobi_name = lobi_name.replace('Lobi:', '')

        return post_detail_list

def downloadMvFile(post_detail_list):

    for i, post_detail in enumerate(post_detail_list):

        #ファイル存在確認　もうあったら落とさない
        if os.path.exists(post_detail.mv_path):
            print('downloadMvFile/動画がもうあるからDLしないよ: %i' % i)
            post_detail_list[i].img_exist = True
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
            print('downloadMvFile/動画DL終わったよ: %i' % i)

    return post_detail_list

def extractCapture(post_detail_list):

    remove_post_details = []
    for i, post_detail in enumerate(post_detail_list):
        
        mv = cv2.VideoCapture(post_detail.mv_path)
        # 動画情報を取得
        fps = mv.get(cv2.CAP_PROP_FPS)
        pos_frame = mv.get(cv2.CAP_PROP_POS_FRAMES)
        cnt_frame = mv.get(cv2.CAP_PROP_FRAME_COUNT)

        # 1.final_score
        # 最終フレームの画像を取得
        post_detail_list[i].cap_scr_f_name = post_detail.id + '_scrF.png'
        post_detail_list[i].cap_scr_f_path = os.path.join(CAPTURE_DIR, post_detail_list[i].cap_scr_f_name)

        pos_frame = cnt_frame
        ret = False
        while not ret:
            #最終フレームから一定間隔(fps)ごとに有効な画像を確認していく
            mv.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)

            #現在フレームの画像を取得
            ret, frame = mv.read()
            if ret == True:
                break
            elif ret == False:
                pos_frame -= fps
                if pos_frame > 0:
                    # print('次の画像見にいくよ: %f' % pos_frame)
                    pass
                else:
                    print('extractCapture/InvalidMovie(This movie file has no frame): %s'
                            % post_detail.mv_name)
                    remove_post_details.append(post_detail)
                    break
                    
        # 有効な画像が取れたら保存する
        cv2.imwrite(post_detail_list[i].cap_scr_f_path, frame)

        # 2.break/nobreak
        # 先頭から15フレーム（0.5秒）毎に取得する
        if int(post_detail.duration.replace(':', '')) > 300:
            pos_frame = 241
        else:
            pos_frame = 181
        j = 0
        ret = True
        while ret and pos_frame > 0:
            # 最終フレームから一定間隔(fps)ごとに有効な画像を確認していく
            mv.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)

            # 現在フレームの画像を取得
            ret, frame = mv.read()
            if ret == True:
                post_detail_list[i].cap_mode_names.append('%s_Mode_%s.png' % (post_detail.id, str(j)))
                post_detail_list[i].cap_mode_paths.append(os.path.join
                                    (CAPTURE_DIR, post_detail_list[i].cap_mode_names[j]))
                cv2.imwrite(post_detail_list[i].cap_mode_paths[j], frame)
            elif ret == False:
                break

            pos_frame -= 15
            j += 1

    # 無効な動画データのpost_detailをリストから除去する
    if len(remove_post_details) > 0:
        for remove_post_detail in remove_post_details:
            print('extractCapture/無効な動画データは対象外処理：%s' % remove_post_detail.id)
            post_detail_list.remove(remove_post_detail)

    return post_detail_list

def ajustCapture(post_detail_list):

    # 1.775 : 568 / 320 iPhone ごま乙の画面サイズ比
    heightGmot = 568
    widthGmot = 320
    aspectGmot = heightGmot / widthGmot 

    for i, post_detail in enumerate(post_detail_list):

        scrFImg = cv2.imread(post_detail.cap_scr_f_path)
        modeImgs = []
        [modeImgs.append(cv2.imread(modeImg)) for modeImg in post_detail.cap_mode_paths]

        height, width = scrFImg.shape[:2]
        height = math.floor(height)
        width = math.floor(width)
        aspect = height / width

        # アスペクト比調整
        if  aspect != aspectGmot:
            #高さに対する適正な幅（補正幅）を求める
            coefficientWidth = height / (aspectGmot * width) 
            correctionWidth = math.floor(width * coefficientWidth)

            #トリミングする位置x1,x2を求める
            halfWidth = math.floor(width / 2) #x軸中央位置
            x1 = int(halfWidth - (correctionWidth / 2))
            x2 = int(halfWidth + (correctionWidth / 2))

            # 2点を通る矩形部分を切り抜き
            # img[y: y + h, x: x + w]
            scrFImg = scrFImg[0:height, x1:x2]

            # Modeについても、同様の計算結果を用いてトリミングする
            for j, modeImg in enumerate(modeImgs):
                modeImgs[j] = modeImg[0:height, x1:x2]

        # 画像サイズ調整
        if not (height == heightGmot and width == widthGmot):
            scrFImg = cv2.resize(scrFImg, (widthGmot, heightGmot))
            for j, modeImg in enumerate(modeImgs):
                modeImgs[j] = cv2.resize(modeImg, (widthGmot, heightGmot))
        else:
            continue
        
        # クリッピングした箇所を保存（上書き）
        cv2.imwrite(post_detail.cap_scr_f_path, scrFImg)
        for j, modeImg in enumerate(modeImgs):
            cv2.imwrite(post_detail.cap_mode_paths[j], modeImg)

        post_detail_list[i].img_edit_flag = True
        
    return post_detail_list

def ocrScore(post_detail_list):

    for i, post_detail in enumerate(post_detail_list):
        
        img = cv2.imread(post_detail.cap_scr_f_path)
        # 画像処理
        img = img[71:91, 35:135]        # 文字領域抽出　矩形[y: y + h, x: x + w]
        img = convGrayImg(img)          # グレースケール変換
        img = adptThreshImg(img, 25)    # 適応的二値変換
        # 各数値領域抽出 x:15+15+15+3(カンマ)+15+15+15
        digit_imgs = [
            img[0:20, 8:22],
            img[0:20, 23:37],
            img[0:20, 38:52],
            img[0:20, 56:70],
            img[0:20, 71:85],
            img[0:20, 86:100]
        ]

        # OCR KNN
        post_detail_list[i].final_score_raw = knnClassify(digit_imgs, KNN_IDENTIFIER_SCORE)
        post_detail_list[i].final_score = int(post_detail_list[i].final_score_raw.replace('_', '0'))

        # debug
        # trmImg output
        # capScrFTrmName = 'trm_' + post_detail.id + '.png'
        # capScrFTrmPath = os.path.join(CAPTURE_DIR, capScrFTrmName)
        # cv2.imwrite(capScrFTrmPath, img)
        # #digitImg output
        # for num, digitImg in enumerate(digitImgs):
        #     digitImgName = str(num) + '_' + post_detail.id + '.png'
        #     digitImgPath = os.path.join(CAPTURE_DIR, digitImgName)
        #     cv2.imwrite(digitImgPath, digitImg)
        # [print(key + ': ' + str(value)) for key, value in post_detail.__dict__.items()]

    return post_detail_list 

def discernMode(post_detail_list):

    for i, post_detail in enumerate(post_detail_list):
        
        knn_mode_result = []
        for j, capModePath in enumerate(post_detail.cap_mode_paths):

            img = cv2.imread(capModePath)
            # 画像処理
            img = img[195:250, 40:280]      # 文字領域抽出　矩形[y: y + h, x: x + w]
            img = convGrayImg(img)          # グレースケール変換
            img = adptThreshImg(img, 25)    # 適応的二値変換

            # DISCERN KNN
            images = []
            images.append(img)
            knn_mode_result.append(knnClassify(images, KNN_IDENTIFIER_MODE))

            # debug
            # trmImg output
            # cv2.imwrite(capModePath, img)

        # 一つでもブレイクと判別された場合、ブレイクと判定する
        post_detail_list[i].stage_mode = 'b' if 'b' in knn_mode_result else 'n'

        # debug
        # print('id:' + post_detail.id)
        # print('stage_mode:' + post_detail_list[i].stage_mode)

    return post_detail_list 

def convGrayImg(img):
    # グレースケール変換
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    return img

def adptThreshImg(img, threshSubstractConst):
    # 適応的二値変換
    # ADAPTIVE_THRESH_MEAN_C 近傍領域の中央値をしきい値
    # ADAPTIVE_THRESH_GAUSSIAN_C 近傍領域の重み付け平均値をしきい値
    max_pixel = 255
    block_size = 11               
    thresh_substract_const = 25    
    img = cv2.adaptiveThreshold(
        img,
        max_pixel,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,                  #しきい値計算に使用する近傍領域のサイズ
        thresh_substract_const        #計算されたしきい値から引く定数
    )

    return img

def sharpenImg(img):
    k = 1.0
    op = np.array([
                    [-k, -k, -k],
                    [-k, 1 + 8 * k, -k],
                    [-k, -k, -k]
                ])
    img = cv2.filter2D(img, -1, op)
    img = cv2.convertScaleAbs(img)

    return img

def insertPosts(post_detail_list):
    Session = sessionmaker(bind=GmotDbAccessor.engine)
    session = Session()
    now = datetime.datetime.now()

    posts = []
    for post in post_detail_list:

        # dataCleanse
        # スコアに認識できない文字が2文字以上ある、または、
        # スコアが400000以上である、または、
        # 動画の長さが1:50未満　のpostは無効データとして扱う
        is_valid_data = '0'
        if (post.final_score_raw.count('_') > 1
            or post.final_score > 400000
            or int(post.duration.replace(':', '')) < 140):
            is_valid_data = '9'
        # スコアが100000を超えるpostはbreakと判断する
        if post.final_score > 100000:
            post.stage_mode = 'b'

        gbPost = GmotDbAccessor.GBPost(
            now,                        # created                 
            '0000-00-00 00:00:00',      # modified                
            post.id,                    # id                      
            post.post_datetime.date(),  # post_date                
            post.meta_ids_name,         # meta_ids_name           
            post.author,                # author
            post.lobi_name,             # lobi_name
            post.user_id,               # user_id                       
            post.bs_att_score,          # bs_att_score     underdev
            post.bs_att_score_raw,      # bs_att_score_raw underdev
            post.final_score,           # final_score
            post.final_score_raw,       # final_score_raw   
            post.stage_mode,            # stage_mode
            post.post_datetime,         # post_datetime           
            post.duration,              # duration                
            post.img_edited,            # is_cap_editted          
            '0',                        # is_final_score_editted
            is_valid_data               # is_valid_data  
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
