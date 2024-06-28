# -----------------------------------------------------------------------
# HashGAN: Deep Learning to Hash with Pair Conditional Wasserstein GAN
# Licensed under The MIT License [see LICENSE for details]
# Modified by Bin Liu
# -----------------------------------------------------------------------
# Based on:
# Improved Training of Wasserstein GANs
# Licensed under The MIT License
# https://github.com/igul222/improved_wgan_training
# -----------------------------------------------------------------------

import argparse
import locale
import os
import sys
import time
from pprint import pprint
from datetime import datetime

import numpy as np
import tensorflow as tf
from easydict import EasyDict
from tqdm import trange

from lib.dataloader import Dataloader
from lib.metric import MAPs
from lib.params import print_param_size, params_with_name
from lib.util import preprocess_resize_scale_img, save_images, scalar_summary
from lib.criterion import cross_entropy
from lib.architecture import generator, discriminator
from lib.config import config, update_and_inference_config


# noinspection PyAttributeOutsideInit
class Model(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = self.cfg.TRAIN.BATCH_SIZE

        self._iteration = tf.placeholder(tf.int32, shape=None)

        self.labeled_real_data_holder = tf.placeholder(tf.int32, shape=[self.batch_size, self.cfg.DATA.OUTPUT_DIM])
        self.unlabeled_real_data_holder = tf.placeholder(tf.int32, shape=[self.batch_size, self.cfg.DATA.OUTPUT_DIM])
        self.labeled_labels_holder = tf.placeholder(tf.int32, shape=[self.batch_size, self.cfg.DATA.LABEL_DIM])
        self.unlabeled_labels_holder = tf.placeholder(tf.int32, shape=[self.batch_size, self.cfg.DATA.LABEL_DIM])

        self.build_graph()
        self.build_fixed_noise_samples()

    def build_graph(self):
        labeled_real_data = self.normalize(self.labeled_real_data_holder)
        unlabeled_real_data = self.normalize(self.unlabeled_real_data_holder)
        labeled_fake_data = generator(self.batch_size, self.labeled_labels_holder, cfg=self.cfg)
        unlabeled_fake_data = generator(self.batch_size, self.unlabeled_labels_holder, cfg=self.cfg)

        # init optimizer
        if self.cfg.TRAIN.DECAY:
            decay = tf.maximum(0., 1. - (tf.cast(self._iteration, tf.float32) / self.cfg.TRAIN.ITERS))
        else:
            decay = 1.0

        all_data = tf.concat([unlabeled_real_data, labeled_real_data, labeled_fake_data, unlabeled_fake_data], axis=0)
        pos_start, pos_middle, pos_end = [i * self.batch_size for i in range(1, 4)]
        disc_wgan_all, disc_acgan_all = discriminator(all_data, cfg=self.cfg)

        # real vs real acgan loss
        self.cost_disc_acgan_rr = cross_entropy(disc_acgan_all[pos_start:pos_middle], self.labeled_labels_holder,
                                                alpha=self.cfg.TRAIN.CROSS_ENTROPY_ALPHA,
                                                normed=self.cfg.TRAIN.NORMED_CROSS_ENTROPY)
        self.cost_disc_acgan = self.cost_disc_acgan_rr
        summary_list_disc = [
                tf.summary.scalar('cost_disc_acgan_rr', self.cost_disc_acgan_rr),
                tf.summary.scalar('cost_disc_acgan', self.cost_disc_acgan)]
        # real vs fake acgan loss, fake can't influence real.
        if self.cfg.TRAIN.ACGAN_SCALE_FAKE != 0:
            self.cost_disc_acgan_fr = cross_entropy(disc_acgan_all[pos_start:pos_middle], self.labeled_labels_holder,
                                                    disc_acgan_all[pos_middle:pos_end], self.labeled_labels_holder,
                                                    alpha=self.cfg.TRAIN.CROSS_ENTROPY_ALPHA,
                                                    normed=self.cfg.TRAIN.NORMED_CROSS_ENTROPY,
                                                    partial=True)
            self.cost_disc_acgan += self.cfg.TRAIN.ACGAN_SCALE_FAKE * self.cost_disc_acgan_fr
            summary_list_disc.append(tf.summary.scalar('cost_disc_acgan_fr', self.cost_disc_acgan_fr))
        self.cost_disc = self.cfg.TRAIN.ACGAN_SCALE * self.cost_disc_acgan
        summary_list_disc.append(tf.summary.scalar('cost_disc', self.cost_disc))

        # disciminator wgan loss
        if self.cfg.TRAIN.WGAN_SCALE != 0.0:
            self.cost_disc_wgan_l = tf.reduce_mean(disc_wgan_all[pos_middle:]) - tf.reduce_mean(disc_wgan_all[:pos_middle])
            self.cost_disc_wgan_gp = self.gradient_penalty(all_data[:pos_middle], all_data[pos_middle:])
            self.cost_disc_wgan = self.cost_disc_wgan_l + self.cfg.TRAIN.WGAN_SCALE_GP * self.cost_disc_wgan_gp
            self.cost_disc += self.cfg.TRAIN.WGAN_SCALE * self.cost_disc_wgan
            summary_list_disc += [
                tf.summary.scalar('cost_disc_wgan_l', self.cost_disc_wgan_l),
                tf.summary.scalar('cost_disc_wgan_gp', self.cost_disc_wgan_gp),
                tf.summary.scalar('cost_disc_wgan', self.cost_disc_wgan)]

        disc_opt = tf.train.AdamOptimizer(learning_rate=self.cfg.TRAIN.LR * decay, beta1=0., beta2=0.9)
        self.train_op_disc = disc_opt.minimize(self.cost_disc, var_list=params_with_name('discriminator'))
        self.gv_disc = disc_opt.compute_gradients(self.cost_disc, var_list=params_with_name('discriminator'))
        self.summary_disc = tf.summary.merge([summary_list_disc])

        # generator loss
        self.gv_gen = []  # TODO: real gv_gen
        if self.cfg.TRAIN.G_LR != 0:
            gen_opt = tf.train.AdamOptimizer(learning_rate=self.cfg.TRAIN.G_LR * decay, beta1=0., beta2=0.9)
            self.cost_gen_wgan = - tf.reduce_mean(disc_wgan_all[pos_middle:])
            self.cost_gen_acgan = self.cost_disc_acgan_fr
            self.cost_gen = self.cfg.TRAIN.WGAN_SCALE_G * self.cost_gen_wgan \
                            + self.cfg.TRAIN.ACGAN_SCALE_G * self.cost_gen_acgan
            self.train_op_gen = gen_opt.minimize(self.cost_gen, var_list=params_with_name('generator'))
            self.gv_gen = gen_opt.compute_gradients(self.cost_gen, var_list=params_with_name('generator'))
            self.summary_gen = tf.summary.merge([
                tf.summary.scalar('cost_gen_wgan', self.cost_gen_wgan),
                tf.summary.scalar('cost_gen_acgan', self.cost_gen_acgan),
                tf.summary.scalar('cost_gen', self.cost_gen),
            ])

        # set acgan_output
        _, self.disc_real_acgan = discriminator(labeled_real_data, stage='val', cfg=self.cfg)

    def build_fixed_noise_samples(self):
        noise_dim = 256 if self.cfg.MODEL.G_ARCHITECTURE == "NORM" else 128
        fixed_noise = tf.constant(np.random.normal(size=(100, noise_dim)).astype('float32'))
        fixed_labels = tf.eye(10, self.cfg.DATA.LABEL_DIM, dtype=tf.int32)
        fixed_labels = tf.reshape(tf.tile(fixed_labels, [1, 10]), (100, self.cfg.DATA.LABEL_DIM))

        self.fixed_noise_samples = generator(100, fixed_labels, noise=fixed_noise, cfg=self.cfg)

    def gradient_penalty(self, real_data, fake_data):
        shape = [2 * self.batch_size, 1]
        reduction_indices = [1]
        if self.cfg.MODEL.D_ARCHITECTURE == "ALEXNET":
            shape += [1, 1]
            reduction_indices += [2, 3]
            real_data = preprocess_resize_scale_img(real_data, width_height=self.cfg.DATA.WIDTH_HEIGHT)
            fake_data = preprocess_resize_scale_img(fake_data, width_height=self.cfg.DATA.WIDTH_HEIGHT)
        alpha = tf.random_uniform(shape=shape, minval=0, maxval=1)

        interpolates = real_data + alpha * (fake_data - real_data)
        gradients = tf.gradients(discriminator(interpolates, cfg=self.cfg)[0], [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=reduction_indices))
        return tf.reduce_mean((slopes - 1.) ** 2)

    @staticmethod
    def normalize(x):
        x = 2 * tf.cast(x, tf.float32) / 256. - 1
        x += tf.random_uniform(shape=x.shape, minval=0., maxval=1. / 128)  # de-quantize
        return x


def forward_all(session, model, data_generator, size, cfg):
    outputs, labels = [], []
    for image, label in data_generator():
        feed_dict = {model.labeled_real_data_holder: image, model.labeled_labels_holder: label}
        outputs.append(session.run(model.disc_real_acgan, feed_dict=feed_dict))
        labels.append(label)
    return EasyDict(output=np.array(outputs).reshape([-1, cfg.MODEL.HASH_DIM])[:size, :],
                    label=np.array(labels).reshape([-1, cfg.DATA.LABEL_DIM])[:size, :])


def evaluate(session, model, dataloader, cfg):
    db = forward_all(session, model, dataloader.db_gen, cfg.DATA.DB_SIZE, cfg)
    test = forward_all(session, model, dataloader.test_gen, cfg.DATA.TEST_SIZE, cfg)
    return MAPs(cfg.DATA.MAP_R).get_maps_by_feature(db, test)


def main(cfg):
    # build graph
    model = Model(cfg)

    # training
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    config_proto.allow_soft_placement = True
    with tf.Session(config=config_proto) as session:
        summary_writer = tf.summary.FileWriter(cfg.DATA.LOG_DIR, session.graph)

        dataloader = Dataloader(cfg.TRAIN.BATCH_SIZE, cfg.DATA.WIDTH_HEIGHT, cfg.DATA.LIST_ROOT, cfg.DATA.DATA_ROOT)
        gen = dataloader.inf_gen(dataloader.train_gen)
        unlabel_gen = dataloader.inf_gen(dataloader.unlabeled_db_gen)

        print_param_size(model.gv_gen, model.gv_disc)

        print("initializing global variables")
        session.run(tf.global_variables_initializer())

        saver_gen = tf.train.Saver(params_with_name('generator'))
        saver_disc = tf.train.Saver(params_with_name('discriminator'))

        if len(cfg.MODEL.G_PRETRAINED_MODEL_PATH) > 0:
            saver_gen.restore(session, cfg.MODEL.G_PRETRAINED_MODEL_PATH)
            print("Generator pretrained model restored: {}".format(cfg.MODEL.G_PRETRAINED_MODEL_PATH))
        if len(cfg.MODEL.D_PRETRAINED_MODEL_PATH) > 0:
            saver_disc.restore(session, cfg.MODEL.D_PRETRAINED_MODEL_PATH)
            print("Discriminator pretrained model restored: {}".format(cfg.MODEL.D_PRETRAINED_MODEL_PATH))

        if cfg.TRAIN.EVALUATE_MODE:
            map_val = evaluate(session, model, dataloader, cfg)
            print('map_val: {}'.format(map_val))
            return 0

        print("training")
        for iteration in trange(cfg.TRAIN.ITERS, desc='Training'):
            start_time = time.time()

            def get_feed_dict():
                labeled_data, labeled_labels = gen()
                unlabeled_data, unlabeled_labels = unlabel_gen()
                return {
                    model.labeled_real_data_holder: labeled_data,
                    model.unlabeled_real_data_holder: unlabeled_data,
                    model.labeled_labels_holder: labeled_labels,
                    model.unlabeled_labels_holder: unlabeled_labels,
                    model._iteration: iteration
                }

            # train generator
            if iteration > 0 and cfg.TRAIN.G_LR != 0:
                summary_gen, _ = session.run([model.summary_gen, model.train_op_gen], feed_dict=get_feed_dict())
                summary_writer.add_summary(summary_gen, iteration)

            # train discriminator
            for i in range(cfg.TRAIN.N_CRITIC):
                summary_disc, _ = session.run([model.summary_disc, model.train_op_disc], feed_dict=get_feed_dict())
                summary_writer.add_summary(summary_disc, iteration * cfg.TRAIN.N_CRITIC + i)

            summary_writer.add_summary(scalar_summary(tag="time", value=time.time() - start_time), iteration)

            # sample images
            if (iteration + 1) % cfg.TRAIN.SAMPLE_FREQUENCY == 0:
                samples = session.run(model.fixed_noise_samples)
                samples = ((samples + 1.) * (255. / 2)).astype('int32')
                save_images(samples.reshape((100, 3, cfg.DATA.WIDTH_HEIGHT, cfg.DATA.WIDTH_HEIGHT)),
                            '{}/samples_{}.png'.format(cfg.DATA.IMAGE_DIR, iteration))

            # calculate mAP score w.r.t all db data_list
            if (iteration + 1) % cfg.TRAIN.EVAL_FREQUENCY == 0 or iteration + 1 == cfg.TRAIN.ITERS:
                map_val = evaluate(session, model, dataloader, cfg)
                print('map_val: {}'.format(map_val))
                summary_writer.add_summary(scalar_summary("mAP_feature", map_val), iteration)

            # save checkpoints
            if (iteration + 1) % cfg.TRAIN.CHECKPOINT_FREQUENCY == 0 or iteration + 1 == cfg.TRAIN.ITERS:
                save_path_gen = os.path.join(cfg.DATA.MODEL_DIR, "G_{}.ckpt".format(iteration))
                save_path_disc = os.path.join(cfg.DATA.MODEL_DIR, "D_{}.ckpt".format(iteration))
                saver_gen.save(session, save_path_gen)
                saver_disc.save(session, save_path_disc)
                print("Model saved in file:")
                print(" - generator: {}".format(save_path_gen))
                print(" - discriminator: {}".format(save_path_disc))


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    locale.setlocale(locale.LC_ALL, '')

    parser = argparse.ArgumentParser(description='HashGAN')
    parser.add_argument('--cfg', '--config', required=True,
                        type=str, metavar="FILE", help="path to yaml config")
    parser.add_argument('--gpus', default='0', type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


    config = update_and_inference_config(args.cfg)
    pprint(config)
    pprint(config, open(os.path.join(config.DATA.OUTPUT_DIR, 'config.txt'), 'w'))

    main(config)

