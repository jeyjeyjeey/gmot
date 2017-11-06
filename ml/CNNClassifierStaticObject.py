# coding: utf-8
import os
import functools
import numpy as np
import cv2
import logging

from keras.applications import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras import metrics, Input

from sklearn.model_selection import train_test_split

from gmot.ml.CNNClassifierKrFTSimpleStackedBase import CNNClassifierKrFTSimpleStackedBase

logger = logging.getLogger(__name__)


def main():
    clsfr = CNNClassifierStaticObject()
    clsfr.train_data_dir = '../traindata/mode_cnn'
    clsfr.intermediate_data_dir = '../features'
    clsfr.weight_dir = '../weight'
    clsfr.log_data_dir = '../log'
    clsfr.identifier = 'mode'
    clsfr.classes = ['b', 'n']

    samples = []
    samples.append(cv2.imread('../traindata/mode/b/Mode_7_f22cc1f5adffd4050dcd8ee722b7ad94b14dd360.png'))
    samples.append(cv2.imread('../traindata/mode/n/Mode_0_efaa00dbc366e7605323d2a0d5b8692ab696fd09.png'))
    samples = clsfr.sample_image(samples, resized_shape=(clsfr.input_x, clsfr.input_y), normalization=True)

    if clsfr.run_train():
        clsfr.prepare_classify()
        y, predictions = clsfr.classify(np.array(samples))
        print(y)
        print(predictions)


class CNNClassifierStaticObject(CNNClassifierKrFTSimpleStackedBase):
    # back_end: tf
    input_x = 244
    input_y = 244
    input_channels = 3
    input_shape_bottom = (input_x, input_y, input_channels)  # channels_last
    input_shape_top = (7, 7, 512)

    def __init__(self):
        super().__init__()

    def _extract_mediate_data(self):
        model = self._get_model_bottom()

        train_data = self.read_entire_data_img(self.classes,
                                               read_mode=cv2.IMREAD_COLOR,
                                               sampling_func=functools.partial(
                                                   self.sample_image,
                                                   resized_shape=(self.input_x, self.input_y),
                                                   normalization=True),
                                               shuffle_flg=True,
                                               one_hot=True)
        X_train, X_validation, y_train, y_validation = train_test_split(
            train_data[0], train_data[1], test_size=0.2, random_state=16)

        features_train = model.predict(X_train)
        np.save(self.intermediate_data_train_file, features_train)
        np.save(self.label_train_file, y_train)

        features_validation = model.predict(X_validation)
        np.save(self.intermediate_data_validation_file, features_validation)
        np.save(self.label_validation_file, y_validation)

        return True

    def _train_top(self):
        start = 0.0015
        stop = 0.001
        epoch = 100

        train_data = np.load(self.intermediate_data_train_file)
        train_labels = np.load(self.label_train_file)
        validation_data = np.load(self.intermediate_data_validation_file)
        validation_labels = np.load(self.label_validation_file)

        model = self._get_model_top()
        sgd = Adam(lr=start, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,
                      metrics=[metrics.mae, metrics.categorical_accuracy])
        learning_rates = np.linspace(start, stop, epoch)
        cb_lr = LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))
        cb_es = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

        hist = model.fit(train_data, train_labels,
                         epochs=epoch,
                         validation_data=(validation_data, validation_labels),
                         callbacks=[cb_lr, cb_es])

        model.save_weights(self.weight_file)
        if not self.log_data_dir == '':
            np.savetxt(os.path.join(self.log_data_dir, 'model_top_vgg_flip_loss.csv'), hist.history['loss'])
            np.savetxt(os.path.join(self.log_data_dir, 'model_top_vgg_flip_val_loss.csv'), hist.history['val_loss'])

        return True

    def _get_model_bottom(self):
        # The bottom model reads the weight here
        if self.model_bottom is None:
            self.model_bottom = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape_bottom)
            # When input_shape isnt defined, ouput_shape also undefined...no wander! then merge error will occur
        return self.model_bottom

    def _get_model_top(self):
        # The top model does not yet reads the weight here. That is why unknown whether before or after training
        if self.model_top is None:
            inputs = Input(self.input_shape_top)
            # _get_model_bottom().output_shape is ok but its stupid to create bottom model just to get the output_shape
            x = Flatten(name='flatten')(inputs)
            x = Dense(256, activation='relu', name='fc1')(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu', name='fc2')(x)
            predictions = Dense(len(self.classes), activation='softmax', name='predictions')(x)
            model = Model(inputs=inputs, outputs=predictions)
            self.model_top = model
        return self.model_top


if __name__ == '__main__':
    main()
