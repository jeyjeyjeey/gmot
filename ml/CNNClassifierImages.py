# coding: utf-8

import numpy as np
import tensorflow as tf
import cv2
import os
import random
from abc import abstractmethod


class CNNClassifierImages():
    _placeholders = {}
    _models = {}
    _is_prepared = False
    summary_writer = None
    summary_op = None

    @abstractmethod
    def prepare(self, sess):
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
    def _run_train(self, sess):
        pass

    @abstractmethod
    def classify(self, sess, imgs):
        pass

    @classmethod
    def get_placeholders(self, key=None):
        return self._placeholders if key is None else self._placeholders.get(key)

    @classmethod
    def get_models(self, key=None):
        return self._models if key is None else self._models.get(key)

    @classmethod
    def set_summary(self, summary_writer, summary_op):
        self.summary_writer = summary_writer
        self.summary_op = summary_op
        return self

    @staticmethod
    # init weight
    # tf.truncated_nomalではランダムな数値で初期化
    # truncated_normal（切断正規分布）とは正規分布の左右を切り取ったもの　重みが0にならないようにしている
    # ランダムにするのは適度にノイズを含めるため（対称性の除去と勾配ゼロ防止のため）
    # ある特徴全てが深い層で消失してしまうことを防ぐ
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    # init bias
    # 0ではなくわずかに陽性=0.1で初期化する
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    @staticmethod
    # convlution
    # x:image
    # W:フィルタ(weight)
    # strides:フィルタの移動幅
    # padding:画像端におけるフィルタ内のpixelがない場合の処理
    #   SAME:0埋めする/VALID:周辺のpixelに定数をかけたものを画素とする？そのため画像サイズが変わる
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    # pooling
    # x:image
    # ksize:窓のサイズ
    # strides:窓の移動幅
    # padding:画像端における窓内のpixelがない場合の処理
    #   SAME:0埋めする/VALID:周辺のpixelに定数をかけたものを画素とする？そのため画像サイズが変わる
    def max_pool(x, size):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                              strides=[1, size, size, 1], padding='SAME')

    @classmethod
    def read_entire_data_img(self, detect_characters, train_data_dir, resize_x=None, resize_y=None, normalization=False):
        samples = None
        labels = None

        for character in detect_characters:
            img_dir = '%s/%s/' % (train_data_dir, str(character))
            files = os.listdir(img_dir)

            # print('read_start:%s' % character)
            # p = ProgressBar().start(len(files), 1)
            for i, file in enumerate(files):
                title, ext = os.path.splitext(file)
                if ext != '.png':
                    continue

                img = cv2.imread(os.path.abspath(img_dir + file),
                                 cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print('Failed to read image')
                    return None
                sample = self._sample_image([img], resize_x, resize_y, normalization)[0]

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

    @classmethod
    def read_iter_data_img(self, detect_characters, train_data_dir,
                           resize_x=None, resize_y=None, normalization=False, shuffle=False):
        samples = None
        labels = None

        for character in detect_characters:
            img_dir = '%s/%s/' % (train_data_dir, str(character))
            files = os.listdir(img_dir)

            if shuffle:
                indeces = random.sample(range(len(files)), len(files))
            else:
                indeces = range(len(files))

            for index in indeces:
                title, ext = os.path.splitext(files[index])
                if ext != '.png':
                    continue

                img = cv2.imread(os.path.abspath(img_dir + files[index]),
                                 cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print('Failed to read image')
                    return None
                sample = self._sample_image([img], resize_x, resize_y, normalization)[0]

                if samples is None:
                    samples = np.empty((0, sample.shape[0] * sample.shape[1]))
                samples = np.append(samples, sample, 0).astype(np.float32)
                if labels is None:
                    labels = np.empty(0)
                labels = np.append(labels, character).astype(np.int)

        labels = np.eye(10)[labels]  # one-hot変換
        data = [samples, labels]

        yield data

    @staticmethod
    def _sample_image(imgs, resize_x=None, resize_y=None, normalization=False):

        img_procs = []
        if resize_x is not None and resize_y is not None:  # サイズ変更
            img_procs.append(lambda img: cv2.resize(img, (resize_x, resize_y)))
        if normalization:  # 正規化
            img_procs.append(lambda img: 1 - np.array(img / 255))
        img_procs.append(lambda img: img.reshape((1, -1)))  # 平坦化

        for img_proc in img_procs:
            imgs = list(map(img_proc, imgs))

        return imgs
