"""
reconstruction cubic using vnet only
OCT 04
"""
from cmath import exp
import tensorflow as tf
import os
import argparse
import numpy as np
import pandas as pd
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.dataloading.data_loader import DataLoader
from core.data.dataloaders import ReconDataloaderVNet
from core.models.models import VNet, count_params
from core.models.transform2_3d import batch_affine_warp3d as spatial_transformer_network_3d
from core.data.datautils import get_list_of_patients, get_split_deterministic
from core.utils.visualizer import DataVisualizer
from core.utils.evaluate import evaluate_single_image
from core.utils.post_processing import post_processing_results2, post_processing_results, errosion, post_eval

from glob import glob
from tqdm import tqdm



class ReconBinTrainer3D(object):

    def __init__(self, args, sess):

        self.args = args

        # build models
        self.lr = tf.placeholder('float')
        self.drop = tf.placeholder('float')

        self.target_surface = tf.placeholder('float', shape=[None, self.args.patch_size, self.args.patch_size, self.args.patch_size, 1])  # semi surface
        self.target_source = tf.placeholder('float', shape=[None, self.args.patch_size, self.args.patch_size, self.args.patch_size, 1])  # full surface, GT

        # vnet
        self.vnet = VNet(self.args.n_filter, self.target_surface, self.drop, self.args.l2)
        self.target_source_prime = self.vnet.create_model()

        # loss function
        self.loss_vnet = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.target_source_prime, labels=self.target_source))

        # optimizer
        self.train_recon_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss_vnet)
        init_op = tf.global_variables_initializer()
        self.init_op = init_op
        count_params()
        self.sess = sess

        self.saver = tf.train.Saver(max_to_keep=25)

        # set exp
        exp_path = os.path.join(self.args.base_path, self.args.exp, f"cv{self.args.cv}_size{self.args.patch_size}_vnet_{self.args.num_samples}")
        self.exp_path = exp_path

        # build_dataset
        self.__init_dataset__(args)

    def get_list_of_patients(self, data_folder):
        patients = []
        target_surface_files = glob(os.path.join(data_folder, "data", "*x2.npy"))
        for target_surface_file in target_surface_files:
            patient_name = target_surface_file[
                                target_surface_file.rfind("/") + 1: target_surface_file.rfind("_")]
            patients.append(patient_name)
        return patients

    def __init_dataset__(self, args):
        data_path = os.path.join(args.base_path, args.data_path, f"{args.data_prefix}_{args.patch_size}_{args.num_samples}")
        patients = self.get_list_of_patients(data_path)
        train_patients, val_patients = get_split_deterministic(patients, fold=args.cv, num_splits=args.cv_max)
        self.train_patients = train_patients
        self.val_patients = val_patients
        # data_path,
        #          data,
        #          batch_size,
        #          patch_size,
        #          num_threads_in_multithreaded,
        dataloader_train = ReconDataloaderVNet(f"{data_path}/data", train_patients, args.batch_size, 
                                        [args.patch_size, args.patch_size, args.patch_size], args.n_workers)
        dataloader_val = ReconDataloaderVNet(f"{data_path}/data", val_patients, args.batch_size, 
                                        [args.patch_size, args.patch_size, args.patch_size], args.n_workers)

        if args.n_workers > 1:
            tr_gen = MultiThreadedAugmenter(dataloader_train, None, num_processes=args.n_workers,
                                            num_cached_per_queue=5, pin_memory=False)
            val_gen = MultiThreadedAugmenter(dataloader_val, None, num_processes=args.n_workers, num_cached_per_queue=5,
                                             pin_memory=False)
            tr_gen.restart()
            val_gen.restart()
        else:
            tr_gen = SingleThreadedAugmenter(dataloader_train, None)
            val_gen = SingleThreadedAugmenter(dataloader_train, None)

        self.tr_gen = tr_gen
        self.val_gen = val_gen
        self.data_path = data_path

    def train(self):
        self.sess.run(self.init_op)
        if not os.path.isdir(self.exp_path):
            os.makedirs(self.exp_path)
        num_train_data = len(self.train_patients)

        #for epoch in tqdm(range(self.args.num_epochs), desc=f"training"):
        for epoch in range(self.args.num_epochs):
            print("[x] epoch: %d, training" % epoch)
            num_batch = num_train_data // self.args.batch_size
            epoch_loss = 0.

            for mini_batch in tqdm(range(num_batch), desc=f"training @ epoch {epoch}", unit="batch"):
                batch_data = next(self.tr_gen)

                target_surface = np.transpose(batch_data['target_surface'], (0, 2, 3, 4, 1))
                target_source = np.transpose(batch_data['target_source'], (0, 2, 3, 4, 1))

                _, train_loss = self.sess.run([self.train_recon_op, self.loss_vnet],
                                               feed_dict={self.target_surface: target_surface,
                                                          self.target_source: target_source,
                                                          self.lr: self.args.lr,
                                                          self.drop: self.args.dropout_p})
                epoch_loss += train_loss

            print("[x] epoch: %d, average loss: %f" % (epoch, epoch_loss / num_batch))

            if (epoch) % self.args.validate_epoch == 0:
                self.test(epoch)
                
    def test(self, epoch):
        print("[x] epoch: %d, validate" % epoch)
        if not os.path.isdir(self.exp_path + "/%04d" % (epoch)):
            os.makedirs(self.exp_path + "/%04d" % (epoch))

        target = open(self.exp_path + "/%04d/val.csv" % (epoch), "w")
        target.write('patient_id,dice,le\n')

        for mini_batch in tqdm(range(len(self.val_patients))):
            patient_id = self.val_patients[mini_batch]
            target_surface = np.load("{}/data/{}_x2.npy".format(self.data_path, patient_id), mmap_mode='r')
            target_source = np.load("{}/data/{}_y.npy".format(self.data_path, patient_id), mmap_mode='r')

            target_surface = np.expand_dims(np.expand_dims(target_surface, axis=3), axis=0).copy()
            target_source_pred = sess.run([self.target_source_prime], feed_dict={self.target_surface: target_surface, self.lr: self.args.lr, self.drop: 1.})
            target_source_pred = np.squeeze(target_source_pred)


            # post processing
            bin, dice_coeff, le = post_processing_results(np.squeeze(target_source_pred),
                                                          np.squeeze(target_source),
                                                          np.squeeze(target_surface))
            if dice_coeff == 0.:
                bin, dice_coeff, le = post_processing_results(np.squeeze(target_source_pred), np.squeeze(target_source), np.squeeze(target_surface), radius=3)
            
            # bin = errosion(bin)
            dice_coeff, le = post_eval(bin, np.squeeze(target_source))

            target.write(f"{patient_id},{dice_coeff},{le}\n")
            dv2 = DataVisualizer(
                [np.squeeze(target_surface), np.squeeze(target_source_pred), bin, np.squeeze(target_source)],
                save_path=self.exp_path + "/%04d/%s_post.png" % (epoch, patient_id))
            dv2.visualize_np(target_source_pred.shape[2], target_source_pred.shape[1], target_source_pred.shape[0])


            np.save(file=self.exp_path + "/%04d/%s_target_source_predicted.npy" % (epoch, patient_id), arr=target_source_pred)
            np.save(file=self.exp_path + "/%04d/%s_target_source_bin.npy" % (epoch, patient_id), arr=bin)
            np.save(file=self.exp_path + "/%04d/%s_target_source_gt.npy" % (epoch, patient_id), arr=target_source)

        target.flush()
        target.close()

        df = pd.read_csv(self.exp_path + "/%04d/val.csv" % (epoch))
        print(f"[x] testing @ epoch {epoch}, dice = {df['dice'].mean()}, le = {df['le'].mean()}")

        self.saver.save(self.sess, os.path.join(self.exp_path, '%04d' % (epoch), 'model.cpkt'))

    def test_noise(self, epoch, gamma):
        print(f"[x] epoch: {epoch}, validate")
        if not os.path.isdir(f"{self.exp_path}/{epoch}"):
            os.makedirs(f"{self.exp_path}/{epoch}")

        target = open(f"{self.exp_path}/{epoch}/val.csv", "w")
        target.write('patient_id,dice,le\n')

        for mini_batch in tqdm(range(len(self.val_patients))):
            patient_id = self.val_patients[mini_batch]
            target_surface = np.load("{}/data/{}_x2.npy".format(self.data_path, patient_id), mmap_mode='r')
            target_surface = self.add_noise(target_surface, mean=0, std=gamma)
            target_source = np.load("{}/data/{}_y.npy".format(self.data_path, patient_id), mmap_mode='r')

            target_surface = np.expand_dims(np.expand_dims(target_surface, axis=3), axis=0).copy()
            target_source_pred = sess.run([self.target_source_prime], feed_dict={self.target_surface: target_surface, self.lr: self.args.lr, self.drop: 1.})
            target_source_pred = np.squeeze(target_source_pred)

            # post processing
            bin, dice_coeff, le = post_processing_results(np.squeeze(target_source_pred),
                                                          np.squeeze(target_source),
                                                          np.squeeze(target_surface))
            if dice_coeff == 0.:
                bin, dice_coeff, le = post_processing_results(np.squeeze(target_source_pred), np.squeeze(target_source), np.squeeze(target_surface), radius=3)
            
            # bin = errosion(bin)
            dice_coeff, le = post_eval(bin, np.squeeze(target_source))

            target.write(f"{patient_id},{dice_coeff},{le}\n")
            dv2 = DataVisualizer(
                [np.squeeze(target_surface), np.squeeze(target_source_pred), bin, np.squeeze(target_source)],
                save_path=f"{self.exp_path}/{epoch}/{patient_id}_post.png")
            dv2.visualize_np(target_source_pred.shape[2], target_source_pred.shape[1], target_source_pred.shape[0])

            np.save(file=f"{self.exp_path}/{epoch}/{patient_id}_target_source_predicted.npy", arr=target_source_pred)
            np.save(file=f"{self.exp_path}/{epoch}/{patient_id}_target_source_bin.npy", arr=bin)
            np.save(file=f"{self.exp_path}/{epoch}/{patient_id}_target_source_gt.npy", arr=target_source)

        target.flush()
        target.close()

        df = pd.read_csv(f"{self.exp_path}/{epoch}/val.csv")
        print(f"[x] testing @ epoch {epoch}, dice = {df['dice'].mean()}, le = {df['le'].mean()}")

    def add_noise(self, x, mean=0, std=1):
        mask = np.zeros_like(x)
        mask[x>0] = 1
        noise = np.random.normal(mean, std, x.shape)
        masked_noise = noise * mask * (np.max(x) - np.min(x))
        noise_array = x + masked_noise
        return noise_array
    
    def load_model(self, model_path):
        print("load model from {}".format(model_path))
        self.saver.restore(self.sess, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--base_path', type=str, default="/media/z/data3/wei")
    parser.add_argument('--exp', type=str, default="exp_vnet")
    parser.add_argument('--num_epochs', type=int, default=201)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--validate_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)

    # cpu
    parser.add_argument('--n_workers', type=int, default=6)

    # gpu
    parser.add_argument('--gpu', type=str, default="0")

    # CV
    parser.add_argument('--cv', type=int, default=0)  # cross validation, CV=5
    parser.add_argument('--cv_max', type=int, default=5)

    # Model and data
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--data_path', type=str, default="data_prepare/MCX")
    parser.add_argument('--data_prefix', type=str, default="shape_single")
    parser.add_argument('--n_filter', type=int, default=32)
    parser.add_argument('--dropout_p', type=float, default=0.8)
    parser.add_argument('--l2', type=float, default=0.2)
    parser.add_argument('--n_cluster', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=1000)


    # train
    parser.add_argument('--train', type=str, default="train")

    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--model_path', type=str, default="/media/z/data3/wei/exp_vnet/cv0_size64_vnet_1000/0140/model.cpkt")


    params = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(params.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    trainer = ReconBinTrainer3D(params, sess)
    if params.train == "train":
        trainer.train()
    elif params.train == "test":
        trainer.load_model(params.model_path)
        trainer.test(9999)
    elif params.train == "robust_test":
        trainer.load_model(params.model_path)
        if params.sigma < 0:
            for sigma in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                trainer.test_noise(f"test_noise_{sigma}", sigma)
        else:
            trainer.test_noise(f"test_noise_{params.sigma}", params.sigma)