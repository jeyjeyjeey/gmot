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
        self.mv_exist = False
        self.is_valid_data = '0'

class PostDetailGuild(PostDetail):
    def __init__(self):
        super().__init__()
        self.final_score = 0
        self.end_score = 0
        self.end_score_raw = ''
        self.total_score = 0
        self.total_score_raw = ''
        self.stage_mode = '9'
        self.media = 'X'
        self.ring = ''
