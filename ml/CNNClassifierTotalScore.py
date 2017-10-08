# coding:utf-8
# Deep MNIST for Experts のほぼパク

import random
import numpy as np
import cv2
import tensorflow as tf
import os
from CNNClassifierImages import CNNClassifierImages
from progressbar import ProgressBar


def main():

    cnn = CNNClassifierTotalScore()

    with tf.Graph().as_default():
        sess = cnn.prepare(tf.Session())
        with sess:
            summary_writer = tf.summary.FileWriter('test/2e-4_0.5/', graph=sess.graph)
            summary_op = tf.summary.merge_all()
            cnn.set_summary(summary_writer, summary_op)

            # test
            test_data = cnn.read_entire_data_img(range(0, 10), './total_score_test', 28, 28, True)
            accuracy_str = cnn.get_models('accuracy').eval(feed_dict={
                cnn.get_placeholders('x'): test_data[0],
                cnn.get_placeholders('y_'): test_data[1],
                cnn.get_placeholders('keep_prob'): 1.0})
            print('test_accurancy: %g' % accuracy_str)

            sess.close()


class CNNClassifierTotalScore(CNNClassifierImages):
    _train_keep_prob = 0.5
    _learning_rate = 1e-4

    @classmethod
    def prepare(self, sess):
        if self._is_prepared:
            print('prepared already')
            return sess

        # placeholder difinition
        self._placeholders['x'] = tf.placeholder('float', shape=[None, 784], name='x')
        self._placeholders['y_'] = tf.placeholder('float', shape=[None, 10], name='y_')
        self._placeholders['keep_prob'] = tf.placeholder('float', name='keep_prob')

        # model definition
        self._models['y_conv'] = self._inference()
        self._models['train'] = self._loss()
        self._models['predition'], self._models['accuracy'] = self._accuracy()

        # run
        sess.run(tf.global_variables_initializer())
        self._run_train(sess)

        self._is_prepared = True

        return sess

    @classmethod
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
            tf.summary.image('input', x_image, 10)  # tensorboard
            # 重みテンソルにバイアスを追加したものをReLU関数に適用(活性化)
            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)

        # 2層 プーリング層1
        # 最大のプールをx_imageに畳み込む
        # 窓の大きさは2x2に設定
        with tf.name_scope('pooling1') as scope:
            h_pool1 = self.max_pool(h_conv1, 2)
            tf.summary.image('h_pool1', tf.reshape(h_pool1, [-1, 14, 14, 1]), 10)  # tensorboard

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
            tf.summary.image('h_pool2', tf.reshape(h_pool2, [-1, 7, 7, 1]), 10)  # tensorboard

        # 5層 密集接続層
        # 2x2で二回畳み込んだので、28/2/2=7*7のサイズの特徴抽出画像になっている
        # またそれらの画像は、同一の画像範囲に対して、64の特徴を抽出した重ね合わせの状態となっている(=*64)
        # バイアスの1024は決め打ちらしい！
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

    @classmethod
    def _loss(self):
        with tf.name_scope('training') as scope:
            # loss
            # 損失関数定義　交差エントロピー
            cross_entropy = -tf.reduce_sum(self.get_placeholders('y_') * tf.log(self.get_models('y_conv')))
            loss_summary = tf.summary.scalar('cross_entropy', cross_entropy)  # tensorboard
            # 最適化クラス定義　確率的勾配降下法（adam法）
            train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(cross_entropy)

        return train_step

    @classmethod
    def _accuracy(self):
        with tf.name_scope('accuracy') as scope:
            y_conv_predition = tf.reduce_max(self.get_models('y_conv'), 1)
            correct_predition = tf.equal(tf.argmax(self.get_models('y_conv'), 1),
                                         tf.argmax(self.get_placeholders('y_'), 1))
            accuracy_op = tf.reduce_mean(tf.cast(correct_predition, 'float'))
            accuracy_summary = tf.summary.scalar('accuracy', accuracy_op)  # tensorboard

        return y_conv_predition, accuracy_op

    @classmethod
    def _run_train(self, sess):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./')
        if ckpt:  # train_reuse
            last_model = ckpt.model_checkpoint_path  # 最後に保存したmodelへのパス
            print('traindata load: ' + last_model)
            saver.restore(sess, last_model)  # 変数データの読み込み
        else:     # train_on
            print('train start')

            # train_data_read
            train_data = self.read_entire_data_img(range(0, 10), './total_score', 28, 28, True)
            index = random.sample(range(len(train_data[0])), len(train_data[0]))

            summary_str = None
            train_times = 1
            correct_times = 0
            # p = ProgressBar().start(len(index), 1)
            for i in index:
                img = [train_data[0][i]]
                label = [train_data[1][i]]

                sess.run(self.get_models('train'), feed_dict={
                    self.get_placeholders('x'): img, self.get_placeholders('y_'): label,
                    self.get_placeholders('keep_prob'): self._train_keep_prob})
                if self.summary_writer is not None and self.summary_op is not None:
                    summary_str = sess.run(self.summary_op, feed_dict={
                        self.get_placeholders('x'): img, self.get_placeholders('y_'): label,
                        self.get_placeholders('keep_prob'): self._train_keep_prob})

                if train_times % 100 == 0:
                    if self.summary_writer is not None and self.summary_op is not None:
                        self.summary_writer.add_summary(summary_str, train_times)
                        self.summary_writer.flush()
                    # 学習パラメータの保存
                    saver.save(sess, 'CNN_total_score.ckpt')

                train_times += 1
                # p.update(train_times)

        return sess

    @classmethod
    def classify(self, sess, imgs):

        if not self._is_prepared:
            print('called before prepare')
            return None

        imgs = self._sample_image(imgs, resize_x=28, resize_y=28, normalization=True)

        chrs = ''
        for img in imgs:
            feed_dict = {
                self.get_placeholders('x'): img,
                self.get_placeholders('y_'): [[0.0] * 10],
                self.get_placeholders('keep_prob'): 1.0}
            p = sess.run(self.get_models('y_conv'), feed_dict=feed_dict)[0]
            guess_label = np.argmax(p)
            predition = self.get_models('predition').eval(session=sess, feed_dict=feed_dict)

            print('guess_label:%s predition:%g' % (guess_label, predition))
            if predition >= 0.9:
                chrs += chr
            else:
                chrs += '_'

        return chrs


if __name__ == '__main__':
    main()
