# coding: utf-8
"""
ごまおつデータモデル
"""

class PostList:
    def __init__(self):
        self.meta_ids_name = ''
        self.rows = ''
        self.posts = ''

class PostDetail:
    def __init__(self):
        self.title = ''
        self.post_url = ''
        self.mv_url = ''
        self.id = ''
        self.author = ''
        self.user_id = ''
        self.post_datetime = ''
        self.duration = ''
        self.meta_ids_name = ''
        self.mv_name = ''
        self.mv_path = ''
        self.cap_scr_f_name = ''
        self.cap_scr_f_path = ''
        self.cap_mode_names = []
        self.cap_mode_paths = []
        self.bs_att_score = 0
        self.bs_att_score_raw = ''
        self.final_score = 0
        self.final_score_raw = ''
        self.stage_mode = '9'
        self.img_edited = False
        self.img_exist = False
