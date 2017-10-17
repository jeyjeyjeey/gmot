# coding: utf-8
"""
gmot data model
"""


class PostList:
    def __init__(self):
        self.meta_ids_name = str()
        self.rows = str()
        self.posts = str()


class PostDetail:
    def __init__(self):
        self.title = str()
        self.post_url = str()
        self.mv_url = str()
        self.id = str()
        self.author = str()
        self.user_id = str()
        self.post_datetime = str()
        self.duration = str()
        self.meta_ids_name = str()
        self.mv_name = str()
        self.mv_path = str()
        self.mv_exist = False
        self.is_valid_data = '0'


class PostDetailGuild(PostDetail):
    def __init__(self):
        super().__init__()
        self.final_score = int()
        self.end_score = int()
        self.end_score_raw = str()
        self.end_score_prediction = None  # np.array((0, 6), dtype=np.float32)
        self.total_score = int()
        self.total_score_raw = str()
        self.total_score_prediction = None  # np.array((0, 6), dtype=np.float32)
        self.stage_mode = '9'
        self.media = 'X'
        self.ring = str()
