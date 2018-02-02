# coding:utf-8
# Deep MNIST for Experts のほぼパク

import os
import logging
import functools
import numpy as np
import tensorflow as tf

from gmot.ml.CNNClassifierTfBase import CNNClassifierTfBase

logger = logging.getLogger(__name__)


def main():

    cnn = CNNClassifierDigit()

    with tf.Graph().as_default():

            # train
            cnn.train_data_dir = '../traindata/total_score_cnn'
            cnn.ckpt_dir = '../ckpt/total_score'
            cnn.prepare_sess_run()

            # graph
            summary_writer = tf.summary.FileWriter('../tensorboard/total_score/%s_%s/'
                                                   % (cnn._learning_rate, cnn._train_keep_prob),
                                                   graph=cnn.sess.graph)
            summary_op = tf.summary.merge_all()
            cnn.set_summary(summary_writer, summary_op)

            cnn.run_train()

            # test
            cnn.train_data_dir = '../traindata/total_score_knn'
            test_data_itr = cnn.read_iter_data_img(list(range(0, 10)),
                                                   sampling_func=functools.partial(
                                                       CNNClassifierTfBase.sample_image,
                                                       resized_shape=(28, 28),
                                                       normalization=True,
                                                       flattening=True),
                                                   shuffle_flg=True,
                                                   batch_size=100)
            test_data = next(test_data_itr)
            accuracy_str = cnn.get_models('accuracy').eval(session=cnn.sess, feed_dict={
                cnn.get_placeholders('x'): test_data[0],
                cnn.get_placeholders('y_'): test_data[1],
                cnn.get_placeholders('keep_prob'): 1.0})
            logger.debug('test_accurancy: %g' % accuracy_str)

            cnn.close_sess()


class CNNClassifierDigit(CNNClassifierTfBase):
    _placeholders = {}
    _models = {}
    _is_model_defined = False

    @classmethod
    def get_placeholders(self, key=None):
        return CNNClassifierDigit._placeholders if key is None else CNNClassifierDigit._placeholders.get(key)

    @classmethod
    def get_models(self, key=None):
        return CNNClassifierDigit._models if key is None else CNNClassifierDigit._models.get(key)

    @classmethod
    def _set_placeholder(self, key, value):
        CNNClassifierDigit._placeholders[key] = value

    @classmethod
    def _set_model(self, key, value):
        CNNClassifierDigit._models[key] = value

    def __init__(self, _train_keep_prob=0.5, _learning_rate=1e-4):
        super().__init__()
        self._train_keep_prob = _train_keep_prob
        self._learning_rate = _learning_rate

    def prepare_sess_run(self):

        if self.sess is not None:
            logger.warning('already prepared')

        # session start
        self.sess = tf.Session()

        if not CNNClassifierDigit._is_model_defined:
            # placeholder definition
            self._set_placeholder('x', tf.placeholder('float', shape=[None, 784], name='x'))
            self._set_placeholder('y_', tf.placeholder('float', shape=[None, 10], name='y_'))
            self._set_placeholder('keep_prob', tf.placeholder('float', name='keep_prob'))

            # model definition
            self._set_model('y_conv', self._inference())
            self._set_model('train', self._loss())
            prediction, accuracy = self._accuracy()
            self._set_model('prediction', prediction)
            self._set_model('accuracy', accuracy)

            CNNClassifierDigit._is_model_defined = True

        # run
        self.sess.run(tf.global_variables_initializer())

    def _inference(self):
        # 1層 畳み込み層1
        # 5x5のフィルタにて32の特徴を計算
        # フィルタサイズ、入力チャンネル数、出力チャンネル（バイアスと同値）
        with tf.name_scope('convolution1') as scope:
            W_conv1 = self.weight_variable([5, 5, 1, 32], 'W_conv1')
            b_conv1 = self.bias_variable([32], 'b_conv1')
            # xを4次元テンソルに作り変える
            # [-1:画像数, 28:横pixel, 28:縦pixel, 1:ピクセルあたりの情報量（グレースケール明度なので1)]
            x_image = tf.reshape(self.get_placeholders('x'), [-1, 28, 28, 1])
            # tf.summary.image('input', x_image, 10)  # tensorboard
            # 重みテンソルにバイアスを追加したものをReLU関数に適用(活性化)
            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)

        # 2層 プーリング層1
        # 最大のプールをx_imageに畳み込む
        # 窓の大きさは2x2に設定
        with tf.name_scope('pooling1') as scope:
            h_pool1 = self.max_pool(h_conv1, 2)
            # tf.summary.image('h_pool1', tf.reshape(h_pool1, [-1, 14, 14, 1]), 10)  # tensorboard

        # 3層 畳み込み2
        # 5x5のフィルタにて64の特徴を計算
        # フィルタサイズ、入力チャンネル数（1層の出力）、出力チャンネル（バイアスと同値）
        with tf.name_scope('convolution2') as scope:
            W_conv2 = self.weight_variable([5, 5, 32, 64], 'W_conv2')
            b_conv2 = self.bias_variable([64], 'b_conv2')
            # 重みテンソルにバイアスを追加したものをReLU関数に適用(活性化)
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)

        # 4層 プーリング層2
        # 最大のプールをx_imageに畳み込む
        # 窓の大きさは2x2に設定
        with tf.name_scope('pooling2') as scope:
            h_pool2 = self.max_pool(h_conv2, 2)
            # tf.summary.image('h_pool2', tf.reshape(h_pool2, [-1, 7, 7, 1]), 10)  # tensorboard

        # 5層 密集接続層
        # 2x2で二回畳み込んだので、28/2/2=7*7のサイズの特徴抽出画像になっている
        # またそれらの画像は、同一の画像範囲に対して、64の特徴を抽出した重ね合わせの状態となっている(=*64)
        # バイアスの1024は決め打ちらしい
        with tf.name_scope('fullconnection1') as scope:
            W_fc1 = self.weight_variable([7 * 7 * 64, 1024], 'W_fc1')
            b_fc1 = self.bias_variable([1024], 'b_fc1')
            # reshapeにてベクトルに戻す
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            # 行列演算と重みを加算し活性化させる
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # 6層 ドロップアウト層
        # アンサンブル平均をとる
        # keep_prob:過学習を防ぐために一定割合の特徴データは捨てる
        # 　トレーニング中は1未満に設定、実際の判別時には全数使用する
        with tf.name_scope('dropout') as scope:
            h_fc1_drop = tf.nn.dropout(h_fc1, self.get_placeholders('keep_prob'))

        # 7層 出力層
        # ソフトマックス回帰
        with tf.name_scope('readout') as scope:
            W_fc2 = self.weight_variable([1024, 10], 'W_fc2')
            b_fc2 = self.bias_variable([10], 'b_fc2')
            # ソフトマックス回帰による正規化
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        return y_conv

    def _loss(self):
        with tf.name_scope('training') as scope:
            # loss
            # 損失関数定義　交差エントロピー
            cross_entropy = -tf.reduce_sum(self.get_placeholders('y_') * tf.log(self.get_models('y_conv')))
            loss_summary = tf.summary.scalar('cross_entropy', cross_entropy)  # tensorboard
            # 最適化クラス定義　確率的勾配降下法（adam法）
            train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(cross_entropy)

        return train_step

    def _accuracy(self):
        with tf.name_scope('accuracy') as scope:
            y_conv_prediction = tf.reduce_max(self.get_models('y_conv'), 1)
            correct_prediction = tf.equal(tf.argmax(self.get_models('y_conv'), 1),
                                          tf.argmax(self.get_placeholders('y_'), 1))
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            accuracy_summary = tf.summary.scalar('accuracy', accuracy_op)  # tensorboard

        return y_conv_prediction, accuracy_op

    def run_train(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt:  # train_reuse
            last_model = ckpt.model_checkpoint_path  # 最後に保存したmodelへのパス
            logger.info('traindata load: ' + last_model)
            saver.restore(self.sess, last_model)  # 変数データの読み込み
        else:     # train_on
            logger.info('train start')

            # train_data_read
            batch_size = 100
            train_data_itr = self.read_iter_data_img(list(range(0, 10)),
                                                     sampling_func=functools.partial(
                                                         CNNClassifierTfBase.sample_image,
                                                         resized_shape=(28, 28),
                                                         normalization=True, flatten=True),
                                                     shuffle_flg=True,
                                                     batch_size=batch_size)

            summary_str = None
            i = 0
            while True:
                try:
                    train_data = next(train_data_itr)
                except(StopIteration):
                    break

                imgs = train_data[0]
                labels = train_data[1]
                train_data_size = len(imgs)

                accuracy_str = self.get_models('accuracy').eval(session=self.sess, feed_dict={
                    self.get_placeholders('x'): imgs,
                    self.get_placeholders('y_'): labels,
                    self.get_placeholders('keep_prob'): 1.0})
                self.sess.run(self.get_models('train'), feed_dict={
                    self.get_placeholders('x'): imgs, self.get_placeholders('y_'): labels,
                    self.get_placeholders('keep_prob'): self._train_keep_prob})
                if self.summary_writer is not None and self.summary_op is not None:
                    summary_str = self.sess.run(self.summary_op, feed_dict={
                        self.get_placeholders('x'): imgs, self.get_placeholders('y_'): labels,
                        self.get_placeholders('keep_prob'): self._train_keep_prob})

                if i % 100 == 0:
                    if self.summary_writer is not None and self.summary_op is not None:
                        self.summary_writer.add_summary(summary_str, i)
                        self.summary_writer.flush()

                i += train_data_size
                logger.info('train_times:%s accuracy:%g' % (i, accuracy_str))

                del train_data, imgs, labels

            # 学習パラメータの保存
            saver.save(self.sess, os.path.join(self.ckpt_dir, 'CNN_Digit.ckpt'))

        return self

    def classify(self, imgs, prediction_filtering=True, prediction_threshold=0.84):

        if self.sess is None:
            logger.warning('called before prepare')
            return None

        imgs = self.sample_image(imgs, resized_shape=(28, 28), normalization=True, flattening=True)

        chrs = ''
        prediction_list = []
        # for img in imgs:
        feed_dict = {
            self.get_placeholders('x'): imgs,
            self.get_placeholders('keep_prob'): 1.0}
        result_predictions = self.sess.run(self.get_models('y_conv'), feed_dict=feed_dict)

        for prediction in result_predictions:
            guess_label = np.argmax(prediction)
            prediction = np.max(prediction)
            # prediction = self.get_models('prediction').eval(session=self.sess, feed_dict=feed_dict)

            logger.debug('guess_label:%s prediction:%g' % (guess_label, prediction))
            if prediction_filtering:
                if prediction >= prediction_threshold:
                    chrs += str(guess_label)
                else:
                    chrs += '_'
            else:
                chrs += str(guess_label)

            prediction_list.append(prediction)

        return chrs, prediction_list


if __name__ == '__main__':
    main()
