# coding: utf-8

import inspect
import os
import logging
from abc import abstractmethod
import numpy as np

from keras.models import Model

from gmot.ml.ClassifierImageLoaderBase import ClassifierImageLoaderBase

logger = logging.getLogger(__name__)


class CNNClassifierKrFTSimpleStackedBase(ClassifierImageLoaderBase):
    identifier_list = []

    def __init__(self):
        super().__init__()
        self.__intermediate_data_dir = './intermediate_data'
        self.__weight_dir = './weight'
        self.__log_data_dir = './log'
        self.__train_data_dir = str()
        self.__identifier = str()
        self.__classes = list()
        self.__model_top = None
        self.__model_bottom = None
        self.__model_full = None

    # business methods structure
    def run_train(self):
        if not (self._is_exist_intermediate_files or os.path.exists(self.weight_file)):
            self._extract_mediate_data()
        if not os.path.exists(self.weight_file):
            self._train_top()
        return True

    @abstractmethod
    def _extract_mediate_data(self):
        pass

    @abstractmethod
    def _train_top(self):
        pass

    def prepare_classify(self):
        self._get_model_full()
        self._load_weight_model_top()
        return True

    def classify(self, imgs):
        """
        :param imgs: type=numpy.ndarray, shape=(batch_size, x, y, channels)
        :return: type=numpy.ndarray, shape=(batch_size, classes)
        """
        return self._decode_predictions(self._get_model_full().predict_on_batch(imgs))

    def _decode_predictions(self, predictions):
        return [self.get_classes(int(np.argmax(pdct))) for pdct in predictions], predictions

    @abstractmethod
    def _get_model_bottom(self):
        pass

    @abstractmethod
    def _get_model_top(self):
        pass

    def _load_weight_model_top(self):
        self.model_top.load_weights(self.weight_file)

    def _get_model_full(self):
        if self.model_full is None:
            self.model_full = Model(inputs=self._get_model_bottom().input,
                                    outputs=self._get_model_top()(self._get_model_bottom().output))
        return self.model_full

    # properties
    # abstractmethod
    @property
    def intermediate_data_dir(self):
        return self.__intermediate_data_dir

    @intermediate_data_dir.setter
    def intermediate_data_dir(self, path):
        if not os.path.exists(path):
            logger.error('Directory not found (%s)' % inspect.currentframe().f_code.co_name)
            raise FileNotFoundError
        self.__intermediate_data_dir = path

    @property
    def intermediate_data_train_file(self):
        return os.path.join(self.intermediate_data_dir, '%s_data_train.npy' % self.identifier)

    @property
    def label_train_file(self):
        return os.path.join(self.intermediate_data_dir, '%s_label_train.npy' % self.identifier)

    @property
    def intermediate_data_validation_file(self):
        return os.path.join(self.intermediate_data_dir, '%s_data_validation.npy' % self.identifier)

    @property
    def label_validation_file(self):
        return os.path.join(self.intermediate_data_dir, '%s_label_validation.npy' % self.identifier)

    def _is_exist_intermediate_files(self):
        for path in [self.intermediate_data_train_file,
                     self.label_train_file,
                     self.intermediate_data_validation_file,
                     self.label_validation_file]:
            if not os.path.exists(path):
                return False
        return True

    @property
    def weight_dir(self):
        return self.__weight_dir

    @weight_dir.setter
    def weight_dir(self, path):
        if not os.path.exists(path):
            logger.error('Directory not found (%s)' % inspect.currentframe().f_code.co_name)
            raise FileNotFoundError
        self.__weight_dir = path

    @property
    def weight_file(self):
        return os.path.join(self.weight_dir, '%s_model_top_vgg.h5' % self.identifier)

    @property
    def log_data_dir(self):
        return self.__log_data_dir

    @log_data_dir.setter
    def log_data_dir(self, path):
        if not os.path.exists(path):
            logger.error('Directory not found (%s)' % inspect.currentframe().f_code.co_name)
            raise FileNotFoundError
        self.__log_data_dir = path

    @property
    def train_data_dir(self):
        return self.__train_data_dir

    @train_data_dir.setter
    def train_data_dir(self, path):
        if not os.path.exists(path):
            logger.error('Directory not found (%s)' % inspect.currentframe().f_code.co_name)
            raise FileNotFoundError
        self.__train_data_dir = path

    @property
    def identifier(self):
        return self.__identifier

    @identifier.setter
    def identifier(self, val):
        if not CNNClassifierKrFTSimpleStackedBase._is_exist_identifier(val):
            logger.error('Identifier duplicated (%s)' % inspect.currentframe().f_code.co_name)
            raise ValueError
        self.__identifier = val
        CNNClassifierKrFTSimpleStackedBase.identifier_list.append(list)

    @classmethod
    def _is_exist_identifier(cls, val):
        return not (val in CNNClassifierKrFTSimpleStackedBase.identifier_list)

    @property
    def classes(self):
        return self.__classes

    @classes.setter
    def classes(self, val):
        if not isinstance(val, list):
            logger.error('Only built-in List can be set for classes (%s)' % inspect.currentframe().f_code.co_name)
            raise ValueError
        self.__classes = val

    def get_classes(self, idx=None):
        return self.__classes if idx is None else self.__classes[idx]

    @property
    def model_top(self):
        return self.__model_top

    @model_top.setter
    def model_top(self, model):
        self.__model_top = model

    @property
    def model_bottom(self):
        return self.__model_bottom

    @model_bottom.setter
    def model_bottom(self, model):
        self.__model_bottom = model

    @property
    def model_full(self):
        return self.__model_full

    @model_full.setter
    def model_full(self, model):
        self.__model_full = model
