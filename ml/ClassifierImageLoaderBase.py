# coding: utf-8

import os
import logging
import inspect
import random
import functools

import numpy as np
from sklearn.utils import shuffle
import cv2

logger = logging.getLogger(__name__)


class ClassifierImageLoaderBase:
    def __init__(self):
        self.__train_data_dir = './train_data'

    @property
    def train_data_dir(self):
        return self.__train_data_dir

    @train_data_dir.setter
    def train_data_dir(self, path):
        if not os.path.exists(path):
            logger.error('Directory not found (%s)' % inspect.currentframe().f_code.co_name)
            raise FileNotFoundError
        self.__train_data_dir = path

    # 一気にメモリにのせるやつ
    # データは、ディレクトリごとに各ラベルの画像が格納されており、ディレクトリ名がラベルとなっている状態を想定
    def read_entire_data_img(self, detect_characters: list, read_mode=cv2.IMREAD_GRAYSCALE,
                             sampling_func=None, shuffle_flg=True, one_hot=True):

        samples = []
        labels = []
        for character in detect_characters:
            img_dir = '%s/%s/' % (self.train_data_dir, str(character))
            files = os.listdir(img_dir)

            # p = ProgressBar().start(len(files), 1)
            for i, file in enumerate(files):
                title, ext = os.path.splitext(file)
                if ext != '.png':
                    continue
                sample = cv2.imread(os.path.abspath(img_dir + file),
                                    read_mode)
                if sample is None:
                    logger.error('Failed to read image')
                    return None
                samples.append(sample)
                labels.append(character)
                # p.update(i + 1)

        if sampling_func is not None:
            samples = sampling_func(samples)
        samples = np.array(samples)
        labels = np.array(labels)
        if one_hot:  # convert to one-hot
            labels = np.eye(len(detect_characters))[[list(map(str, detect_characters)).index(l) for l in labels]]

        if shuffle_flg:
            data = shuffle(samples, labels, random_state=42)
        else:
            data = [samples, labels]

        return data

    # バッチサイズごとに逐次読み出すやつ（ジェネレータ）
    # データは、ディレクトリごとに各ラベルの画像が格納されており、ディレクトリ名がラベルとなっている状態を想定
    def read_iter_data_img(self, detect_characters: list, read_mode=cv2.IMREAD_GRAYSCALE,
                           sampling_func=None, shuffle_flg=True, one_hot=True, batch_size=100):

        label_dirs = os.listdir(self.train_data_dir)
        sample_files = []
        for label_dir in label_dirs:
            label_dir_path = os.path.join(self.train_data_dir, label_dir)
            if os.path.isdir(label_dir_path) and label_dir in [str(char) for char in detect_characters]:
                files = os.listdir(label_dir_path)
                for file in files:
                    title, ext = os.path.splitext(file)
                    if ext != '.png':
                        continue
                    sample_files.append([file, label_dir])

        if shuffle_flg:
            indices = random.sample(range(len(sample_files)), len(sample_files))
        else:
            indices = range(len(sample_files))

        current_index = 0
        while True:
            if current_index > len(indices):
                raise StopIteration
            proc_indices = indices[current_index:current_index + batch_size]

            samples = []
            labels = []
            for index in proc_indices:
                sample = cv2.imread(os.path.join(self.train_data_dir,
                                    sample_files[index][1], sample_files[index][0]),
                                    read_mode)
                if sample is None:
                    logger.error('Failed to read image')
                    return None
                samples.append(sample)
                labels.append(sample_files[index][1])

            if sampling_func is not None:
                samples = sampling_func(samples)
            samples = np.array(samples)
            labels = np.array(labels)
            if one_hot:  # convert to one-hot
                labels = np.eye(len(detect_characters))[[list(map(str, detect_characters)).index(l) for l in labels]]
                # Using fancy index reference.
                # Specify a list(dtype:Integer) in the indices of ndarray,
                # then it returns ndarray that contains elements of specified indices.

            data = [samples, labels]
            current_index += batch_size

            yield data

    @staticmethod
    def sample_image(imgs,
                     trimmed_shape: tuple=None,
                     resized_shape: tuple=None,
                     normalization=False,
                     flattening=False,
                     expand_img=False,
                     expand_outside=False,
                     other_sample_func_list: list=None):

        img_procs = []
        if trimmed_shape is not None:
            img_procs.append(functools.partial(ClassifierImageLoaderBase.trim, trimmed_shape=trimmed_shape))
        if resized_shape is not None:
            img_procs.append(functools.partial(ClassifierImageLoaderBase.resize, resized_shape=resized_shape))
        if normalization:
            img_procs.append(ClassifierImageLoaderBase.normalize)
        if flattening:
            img_procs.append(ClassifierImageLoaderBase.flatten)
        if expand_img:                  # 各画像を入れ子
            img_procs.append(ClassifierImageLoaderBase.expand_outside)
        if other_sample_func_list is not None:
            img_procs.extend(other_sample_func_list)

        for img_proc in img_procs:
            imgs = list(map(img_proc, imgs))
            # ndarrayの場合はaxis=1要素に対して適用するみたいなので、build-in Listでもどっちでも可

        if expand_outside:              # 外側を入れ子
            imgs = ClassifierImageLoaderBase.expand_outside(imgs)

        return imgs

    @staticmethod
    # Extract area(rectangular[y: y + h, x: x + w])
    def trim(img, trimmed_shape: tuple):
        return img[trimmed_shape[0]:trimmed_shape[1], trimmed_shape[2]:trimmed_shape[3]]

    @staticmethod
    # Resize(x, y)
    def resize(img, resized_shape: tuple):
        return cv2.resize(img, (resized_shape[0], resized_shape[1]))

    @staticmethod
    # Adjust each pixel value to 0 - 1
    def normalize(img):
        return 1 - np.array(img / 255)

    @staticmethod
    # Deform the image dimension to one dimension
    def flatten(img):
        return img.reshape((-1))

    @staticmethod
    # Expand ndarray or built-in List dimension along to axis 0
    def expand_outside(img):
        return np.expand_dims(img, axis=0) if isinstance(img, np.ndarray) else [img]

    @staticmethod
    # Convert color mode
    def convert_channels(img, mode):
        """
        :param img:
        :param mode: main mode...
            cv2.COLOR_BGR2BGRA
            cv2.COLOR_RGB2BGR
            cv2.COLOR_BGR2RGB
            cv2.COLOR_BGR2GRAY
            cv2.COLOR_RGB2GRAY
            cv2.COLOR_GRAY2BGR
            cv2.COLOR_GRAY2RGB
            ...about other mode, see documents
        :return:
        """
        img = cv2.cvtColor(
            img,
            mode
        )
        return img

    @staticmethod
    # Adaptive thresh
    def adapt_thresh(img, thresh_substract_const=25, block_size=11):
        max_pixel = 255
        img = cv2.adaptiveThreshold(
            img,
            max_pixel,
            cv2.ADAPTIVE_THRESH_MEAN_C,     # The median value of the neighboring region is set as a threshold value
            cv2.THRESH_BINARY,
            block_size,                     # Size of eighboring region used for threshold calculation
            thresh_substract_const          # A constant that subtracts the calculated threshold value
        )
        return img
