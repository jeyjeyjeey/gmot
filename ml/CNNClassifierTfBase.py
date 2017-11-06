# coding: utf-8

import logging
from abc import abstractmethod
import tensorflow as tf

from gmot.ml.ClassifierImageLoaderBase import ClassifierImageLoaderBase

logger = logging.getLogger(__name__)


class CNNClassifierTfBase(ClassifierImageLoaderBase):
    def __init__(self):
        super().__init__()
        self.__sess = None
        self.__ckpt_dir = './ckpt'
        self.__summary_writer = None
        self.__summary_op = None

    # business methods structure
    @abstractmethod
    def prepare_sess_run(self):
        pass

    @abstractmethod
    def _inference(self):
        pass

    @abstractmethod
    def _loss(self):
        pass

    @abstractmethod
    def _accuracy(self):
        pass

    @abstractmethod
    def run_train(self):
        pass

    def prepare_classify(self):
        self.prepare_sess_run()
        self.run_train()
        pass

    @abstractmethod
    def classify(self, imgs):
        pass

    def close_sess(self):
        self.sess.close()
        return True

    # properties
    @property
    def sess(self):
        return self.__sess

    @sess.setter
    def sess(self, sess):
        self.__sess = sess

    @property
    def ckpt_dir(self):
        return self.__ckpt_dir

    @ckpt_dir.setter
    def ckpt_dir(self, path):
        self.__ckpt_dir = path

    def set_summary(self, summary_writer, summary_op):
        self.__summary_writer = summary_writer
        self.__summary_op = summary_op

    @property
    def summary_writer(self):
        return self.__summary_writer

    @property
    def summary_op(self):
        return self.__summary_op

    @staticmethod
    # init weight
    # ニューロンの価値の重み
    # tf.truncated_nomalではランダムな数値で初期化
    # truncated_normal（切断正規分布）とは正規分布の左右を切り取ったもの　重みが0にならないようにしている
    # ランダムにするのは適度にノイズを含めるため（対称性の除去と勾配ゼロ防止のため）
    # ある特徴全てが深い層で消失してしまうことを防ぐ
    def weight_variable(shape, name, stddev=0.1):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=name)

    @staticmethod
    # init bias
    # ニューロン発火の閾値
    # 0ではなくわずかに陽性=0.1で初期化する
    def bias_variable(shape, name, bias_value=0.1):
        initial = tf.constant(bias_value, shape=shape)
        return tf.Variable(initial, name=name)

    @staticmethod
    # convlution
    # x:image
    # W:フィルタ(weight)
    # strides:フィルタの移動幅
    # padding:画像端におけるフィルタ内のpixelがない場合の処理
    #   SAME:0埋めする/VALID:周辺のpixelに定数をかけたものを画素とする、そのため画像サイズが変わる
    def conv2d(x, W, padding='SAME'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    @staticmethod
    # pooling
    # x:image
    # ksize:窓のサイズ
    # strides:窓の移動幅
    # padding:画像端における窓内のpixelがない場合の処理
    #   SAME:0埋めする/VALID:周辺のpixelに定数をかけたものを画素とする、そのため画像サイズが変わる
    def max_pool(x, size, padding='SAME'):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                              strides=[1, size, size, 1], padding=padding)
