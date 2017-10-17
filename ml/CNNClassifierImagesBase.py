# coding: utf-8

import os
import logging
import random
from abc import abstractmethod

import numpy as np
import tensorflow as tf
import cv2

logger = logging.getLogger(__name__)


class CNNClassifierImagesBase():
    def __init__(self):
        self._sess = None
        self._train_data_dir = './train_data'
        self._ckpt_dir = './ckpt'
        self.summary_writer = None
        self.summary_op = None

    @abstractmethod
    def prepare(self):
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
    def _run_train(self):
        pass

    @abstractmethod
    def classify(self, imgs):
        pass

    def get_sess(self):
        return self._sess

    def _set_sess(self, sess):
        self._sess = sess

    def set_train_data_dir(self, train_data_dir):
        self._train_data_dir = train_data_dir

    def set_ckpt_dir(self, ckpt_dir):
        self._ckpt_dir = ckpt_dir

    def set_summary(self, summary_writer, summary_op):
        self.summary_writer = summary_writer
        self.summary_op = summary_op

    def get_train_data_dir(self):
        return self._train_data_dir

    def get_ckpt_dir(self):
        return self._ckpt_dir

    def get_summary_writer(self):
        return self.summary_writer

    def get_summary_op(self):
        return self.summary_op

    def close_sess(self):
        self.get_sess().close()
        return True

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

    # 一気にメモリにのせるやつ
    def read_entire_data_img(self, detect_characters, sampling_func=None):
        samples = None
        labels = None

        for character in detect_characters:
            img_dir = '%s/%s/' % (self.get_train_data_dir(), str(character))
            files = os.listdir(img_dir)

            # p = ProgressBar().start(len(files), 1)
            for i, file in enumerate(files):
                title, ext = os.path.splitext(file)
                if ext != '.png':
                    continue

                img = cv2.imread(os.path.abspath(img_dir + file),
                                 cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.error('Failed to read image')
                    return None
                if sampling_func is not None:
                    sample = sampling_func([img])[0]

                if samples is None:
                    samples = np.empty((0, sample.shape[0] * sample.shape[1]))
                samples = np.append(samples, sample, 0).astype(np.float32)
                if labels is None:
                    labels = np.empty(0)
                labels = np.append(labels, character).astype(np.int)

                # p.update(i + 1)

        labels = np.eye(10)[labels]  # one-hot変換
        data = [samples, labels]

        return data

    # バッチサイズごとに逐次読み出すやつ（ジェネレータ）
    # データは、ディレクトリごとに各ラベルの画像が格納されており、ディレクトリ名がラベルとなっている状態を想定
    def read_iter_data_img(self, detect_characters, sampling_func=None, shuffle=True, batch_size=100):
        label_dirs = os.listdir(self.get_train_data_dir())
        sample_files = []
        for label_dir in label_dirs:
            label_dir_path = os.path.join(self.get_train_data_dir(), label_dir)
            if os.path.isdir(label_dir_path) and label_dir in [str(char) for char in detect_characters]:
                files = os.listdir(label_dir_path)
                for file in files:
                    title, ext = os.path.splitext(file)
                    if ext != '.png':
                        continue
                    sample_files.append([file, label_dir])

        if shuffle:
            indeces = random.sample(range(len(sample_files)), len(sample_files))
        else:
            indeces = range(len(sample_files))

        current_index = 0
        while True:
            if current_index > len(indeces):
                raise StopIteration
            proc_indeces = indeces[current_index:current_index + batch_size]

            samples = None
            labels = None
            for index in proc_indeces:
                img = cv2.imread(os.path.join(self.get_train_data_dir(),
                                              sample_files[index][1], sample_files[index][0]),
                                 cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.error('Failed to read image')
                    return None
                if sampling_func is not None:
                    sample = sampling_func([img])[0]

                if samples is None:
                    samples = np.empty((0, sample.shape[0] * sample.shape[1]))
                samples = np.append(samples, sample, 0).astype(np.float32)
                if labels is None:
                    labels = np.empty(0)
                labels = np.append(labels, sample_files[index][1]).astype(np.int)
            labels = np.eye(10)[labels]  # one-hot変換
            data = [samples, labels]
            current_index += batch_size

            yield data

    @staticmethod
    def sample_image(imgs, resize_x=None, resize_y=None, normalization=False):

        img_procs = []
        if resize_x is not None and resize_y is not None:  # サイズ変更
            img_procs.append(lambda img: cv2.resize(img, (resize_x, resize_y)))
        if normalization:  # 正規化
            img_procs.append(lambda img: 1 - np.array(img / 255))
        img_procs.append(lambda img: img.reshape((1, -1)))  # 平坦化

        for img_proc in img_procs:
            imgs = list(map(img_proc, imgs))

        return imgs
