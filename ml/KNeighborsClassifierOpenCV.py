import os
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class KNeighborsClassifierOpenCV:
    dict_knn = {}


class Knn:
    def __init__(self, classifier, k):
        self.classifier = classifier
        self.k = k


def knn_train(detect_characters, train_data_dir, knn_identifier, k):
    samples = None
    labels = []

    for character in detect_characters:
        img_dir = train_data_dir + '/' + str(character) + '/'
        files = os.listdir(img_dir)

        for file in files:
            title, ext = os.path.splitext(file)
            if ext != '.png':
                continue

            abs_path = os.path.abspath(img_dir + file)
            img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error('Failed to read image')
                break

            sample = img.reshape((1, -1))
            label = ord(character)

            if samples is None:
                samples = np.empty((0, img.shape[0] * img.shape[1]))
            samples = np.append(samples, sample, 0).astype(np.float32)
            labels.append(label)

    labels = np.array(labels, np.float32)
    labels = labels.reshape((labels.size, 1)).astype(np.float32)

    knn = cv2.ml.KNearest_create()
    knn.train(samples, cv2.ml.ROW_SAMPLE, labels)

    knn = Knn(knn, k)
    KNeighborsClassifierOpenCV.dict_knn[knn_identifier] = knn

    return True


def knn_classify(images, knn_identifier):

    knn = KNeighborsClassifierOpenCV.dict_knn.get(knn_identifier)
    if knn is None:
        logger.error(knn_identifier + ' is not trained')
        exit(1)

    chr_str = ''
    for image in images:

        sample = image.reshape((1, image.shape[0] * image.shape[1]))
        sample = np.array(sample, np.float32)

        retVal, results, neighResp, dists = knn.classifier.findNearest(sample, knn.k)
        chr_str += chr(int(results.ravel()))

    return chr_str
