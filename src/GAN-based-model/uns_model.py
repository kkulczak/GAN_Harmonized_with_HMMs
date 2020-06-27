from collections import defaultdict
from glob import glob
import time

from torch.utils.tensorboard import SummaryWriter

from lib.discriminator import *
from lib.module import *
from evalution import *

import tensorflow as tf
import numpy as np
import os, sys


def assign_label(x):
    if 'score' in x:
        return f'scores/{x}'
    if 'loss' in x:
        return f'loss/{x}'
    return x


class model(object):
    def __init__(self, config, train=True):
        cout_word = 'UNSUPERVISED MODEL: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('input') as scope:
                # Acoustic Data
                #   Framewise Feature
                self.frame_feat = tf.placeholder(tf.float32, shape=[None,
                                                                    config.feat_max_length,
                                                                    config.feat_dim])
                self.frame_len = tf.placeholder(tf.int32, shape=[None])
                #   Sampled Feature
                self.sample_feat = tf.placeholder(tf.float32, shape=[None,
                                                                     config.phn_max_length,
                                                                     config.feat_dim])
                self.sample_len = tf.placeholder(tf.int32, shape=[None])
                self.sample_rep = tf.placeholder(tf.int32, shape=[None])

                # Real Data
                self.target_idx = tf.placeholder(tf.int32, shape=[None,
                                                                  config.phn_max_length])
                self.target_len = tf.placeholder(tf.int32, shape=[None])

                self.frame_temp = tf.placeholder(tf.float32, shape=[])
                self.global_step = tf.Variable(0, name='global_step',
                                               trainable=False)

            with tf.variable_scope('generator') as scope:
                # Get generated phoneme sequence
                self.fake_sample, _, _ = frame2phn(
                    self.sample_feat,
                    config,
                    self.frame_temp,
                    input_len=self.sample_len,
                    reuse=False
                )

                # Get framewise phoneme distribution
                self.frame_prob, _, _ = frame2phn(
                    self.frame_feat,
                    config,
                    self.frame_temp,
                    input_len=self.frame_len,
                    reuse=True
                )

                # Get framewise prediction
                self.frame_pred = tf.argmax(self.frame_prob, axis=-1)

            with tf.variable_scope('discriminator') as scope:
                # Get real phoneme sequence
                self.real_sample = generate_real_sample(self.target_idx,
                                                        self.target_len,
                                                        config.phn_size)
                inter_sample = generate_inter_sample(self.real_sample,
                                                     self.fake_sample)

                # weak discriminator
                emb = creating_embedding_matrix(
                    config.phn_size,
                    config.dis_emb_size,
                    'emb'
                )
                real_sample_pred = weak_discriminator(
                    self.real_sample,
                    emb,
                    config.dis_hidden_1_size,
                    config.dis_hidden_2_size,
                    reuse=False
                )
                fake_sample_pred = weak_discriminator(
                    self.fake_sample,
                    emb,
                    config.dis_hidden_1_size,
                    config.dis_hidden_2_size,
                    reuse=True
                )
                inter_sample_pred = weak_discriminator(
                    inter_sample,
                    emb,
                    config.dis_hidden_1_size,
                    config.dis_hidden_2_size,
                    reuse=True
                )

                # gradient penalty
                self.gradient_penalty = compute_penalty(inter_sample_pred,
                                                        inter_sample)

            if train:
                self.learning_rate = tf.placeholder(tf.float32, shape=[])

                with tf.variable_scope('segmental_loss') as scope:
                    sep_size = (config.batch_size * config.repeat) // 2
                    self.seg_loss = segment_loss(
                        self.fake_sample[:sep_size],
                        self.fake_sample[sep_size:],
                        repeat_num=self.sample_rep
                    )

                with tf.variable_scope('discriminator_loss') as scope:
                    self.real_score = tf.reduce_mean(real_sample_pred)
                    self.fake_score = tf.reduce_mean(fake_sample_pred)
                    self.dis_loss = (
                        - (self.real_score - self.fake_score)
                        + config.penalty_ratio * self.gradient_penalty
                    )

                with tf.variable_scope('generator_loss') as scope:
                    self.gen_loss = (
                        - (
                            self.fake_score
                           # - self.real_score
                           )
                        # + config.seg_loss_ratio * self.seg_loss
                    )

                self.dis_variables = [v for v in tf.trainable_variables()
                                      if v.name.startswith("discriminator")]
                self.gen_variables = [v for v in tf.trainable_variables()
                                      if v.name.startswith("generator")]

                # Discriminator optimizer
                train_dis_op = tf.train.AdamOptimizer(self.learning_rate,
                                                      beta1=0.5, beta2=0.9)
                gradients = tf.gradients(self.dis_loss, self.dis_variables)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.train_dis_op = train_dis_op.apply_gradients(
                    zip(clipped_gradients, self.dis_variables))

                # Generator optimizer
                train_gen_op = tf.train.AdamOptimizer(self.learning_rate,
                                                      beta1=0.5, beta2=0.9)
                gradients = tf.gradients(self.gen_loss, self.gen_variables)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.train_gen_op = train_gen_op.apply_gradients(
                    zip(clipped_gradients, self.gen_variables))

        sys.stdout.write('\b' * len(cout_word))
        cout_word = 'UNSUPERVISED MODEL: finish     '
        sys.stdout.write(cout_word + '\n')
        sys.stdout.flush()

    def train(self, config, sess, saver, data_loader, dev_data_loader=None,
              aug=False):
        print('TRAINING(unsupervised)...')
        if aug:
            get_target_batch = data_loader.get_aug_target_batch
        else:
            get_target_batch = data_loader.get_target_batch

        batch_size = config.batch_size * config.repeat
        stats = defaultdict(list)
        max_fer = 100.0
        frame_temp = 0.9

        mydir = os.path.join(os.getcwd(), '..', '..', 'data', time.strftime('%Y-%m-%d_%H-%M-%S'))

        os.makedirs(mydir, exist_ok=True)
        writer = SummaryWriter(mydir)

        for step in range(1, config.step + 1):
            if step == 8000:
                frame_temp = 0.8
            if step == 12000:
                frame_temp = 0.7
            for _ in range(config.dis_iter):
                batch_sample_feat, batch_sample_len, batch_repeat_num = \
                    data_loader.get_sample_batch(
                        config.batch_size,
                        repeat=config.repeat
                    )

                batch_target_idx, batch_target_len = get_target_batch(
                    batch_size)

                feed_dict = {
                    self.sample_feat:   batch_sample_feat,
                    self.sample_len:    batch_sample_len,
                    self.target_idx:    batch_target_idx,
                    self.target_len:    batch_target_len,
                    self.learning_rate: config.dis_lr,
                    self.frame_temp:    frame_temp
                }

                run_list = [
                    self.dis_loss,
                    self.train_dis_op,
                    self.real_score,
                    self.fake_score,
                    self.gradient_penalty,
                    self.real_sample,
                    self.fake_sample,
                ]
                (
                    dis_loss,
                    train_dis_op,
                    d_real_score,
                    d_fake_score,
                    d_gradient_penalty,
                    d_real_sample,
                    d_fake_sample,
                ) = sess.run(run_list, feed_dict=feed_dict)
                stats['dis_loss'].append(dis_loss)
                stats['d_real_score'].append(d_real_score)
                stats['d_fake_score'].append(d_fake_score)
                stats['d_gradient_penalty'].append(d_gradient_penalty)
                stats['diff_score'].append(d_fake_score - d_real_sample)

            for _ in range(config.gen_iter):
                batch_sample_feat, batch_sample_len, batch_repeat_num = \
                    data_loader.get_sample_batch(
                        config.batch_size,
                        repeat=config.repeat
                    )
                batch_target_idx, batch_target_len = get_target_batch(
                    batch_size)

                feed_dict = {
                    self.sample_feat:   batch_sample_feat,
                    self.sample_len:    batch_sample_len,
                    self.target_idx:    batch_target_idx,
                    self.target_len:    batch_target_len,
                    self.sample_rep:    batch_repeat_num,
                    self.learning_rate: config.gen_lr,
                    self.frame_temp:    frame_temp
                }

                run_list = [
                    self.gen_loss,
                    self.seg_loss,
                    self.train_gen_op,
                    self.fake_sample,
                    self.real_score,
                    self.fake_score,
                ]
                (
                    gen_loss,
                    seg_loss,
                    train_gen_op,
                    g_fake_sample,
                    g_real_score,
                    g_fake_score,
                ) = sess.run(run_list, feed_dict=feed_dict)

                stats['gen_loss'].append(gen_loss)
                stats['seg_loss'].append(seg_loss)
                stats['g_real_score'].append(g_real_score)
                stats['g_fake_score'].append(g_fake_score)

            if step % config.print_step == 0:
                for k in stats.keys():
                    stats[k] = np.array(stats[k]).mean()

                print(
                    f'Step: {step:5d} \n'
                    f'dis_loss: {stats["dis_loss"]:.4f} '
                    f'== fake:{stats["d_fake_score"]:.4f} '
                    f'- real:{stats["d_real_score"]:.4f} '
                    f'+ {config.penalty_ratio:.1f} '
                    f'* gp:{stats["d_gradient_penalty"]:.4f}'
                )
                print(
                    f'gen_loss: {stats["gen_loss"]:.4f} '
                    f'== real:{stats["g_real_score"]:.4f} '
                    f'- fake:{stats["g_fake_score"]:.4f} '
                    f'+ {config.seg_loss_ratio:.1f} * seg_loss:'
                    f'{stats["seg_loss"]:.4f}'
                )
                for k, v in stats.items():
                    writer.add_scalar(
                        assign_label(k),
                        v,
                        global_step=step,
                    )
                    writer.flush()

                stats.clear()

            if step % config.eval_step == 0:
                step_fer = frame_eval(sess, self, dev_data_loader)
                print(f'EVAL max: {max_fer:.2f} step: {step_fer:.2f}')
                writer.add_scalar(
                    'acc/fer',
                    100.0 - step_fer,
                    global_step=step
                )
                if step_fer < max_fer:
                    max_fer = step_fer
                    saver.save(sess, config.save_path)

        print('=' * 80)
