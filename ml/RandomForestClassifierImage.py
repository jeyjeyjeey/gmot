# coding: utf-8
import logging
import functools

from sklearn.ensemble import RandomForestClassifier

from gmot.ml.ClassifierImageLoaderBase import ClassifierImageLoaderBase

logger = logging.getLogger(__name__)


def main():
    TRAIN_DATA_DIR_MODE = '../gmot/traindata/mode_cnn'
    clf = RandomForestClassifierImage()
    clf.classes = ['n', 'b']
    clf.train_data_dir = TRAIN_DATA_DIR_MODE


class RandomForestClassifierImage(ClassifierImageLoaderBase):
    def __init__(self):
        super().__init__()
        __classifier = None
        __classes = list()

    @property
    def classifier(self):
        return self.__classifier

    @classifier.setter
    def classifier(self, clf):
        self.__classifier = clf

    @property
    def classes(self):
        return self.__classes

    @classes.setter
    def classes(self, val):
        self.__classes = val

    def prepare_classify(self):
        if self.classifier is not None:
            logger.warning('already prepared')
            return False

        train_data = self.read_entire_data_img(self.classes,
                                               sampling_func=functools.partial(
                                                   self.sample_image,
                                                   normalization=True,
                                                   flattening=True),
                                               shuffle_flg=True,
                                               one_hot=False
                                               )

        # n_estimators:number of dicision trees
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(train_data[0], train_data[1])

        self.classifier = clf

        return True

    def classify(self, imgs):
        if self.classifier is None:
            logger.warning('called before prepare')
            return None

        return self.classifier.predict(imgs)


if __name__ == '__main__':
    main()
