"""
reconstruction cubic

final model
June 2023
"""
import tensorflow as tf
import os
import scipy.io as scio
import argparse
import numpy as np
import pandas as pd
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.dataloading.data_loader import DataLoader
from core.models.models import VNet, count_params, STN
from core.models.transform2_3d import batch_affine_warp3d as spatial_transformer_network_3d
from core.data.datautils import get_list_of_patients, get_split_deterministic
from core.utils.visualizer import DataVisualizer
from core.utils.evaluate import evaluate_single_image
from core.utils.post_processing import post_processing_results2, post_processing_results, errosion, post_eval
from core.utils.post_processing import post_processing_results_dual
from core.utils.evaluate import threshold_by_otsu, dice_coefficient_in_train

from glob import glob
from tqdm import tqdm

from core.data.dataloaders import ReconDataloader

class ReconBinTrainer3D(object):

    def __init__(self, args, sess):

        self.args = args

        # build models
        self.lr = tf.placeholder('float')
        self.drop = tf.placeholder('float')

        self.template_surface = tf.placeholder('float', shape=[None, self.args.patch_size, self.args.patch_size, self.args.patch_size, 1])  # template semi surface
        self.template_source = tf.placeholder('float', shape=[None, self.args.patch_size, self.args.patch_size, self.args.patch_size, 1])  # template full surface
        self.target_surface = tf.placeholder('float', shape=[None, self.args.patch_size, self.args.patch_size, self.args.patch_size, 1])  # semi surface
        self.target_source = tf.placeholder('float', shape=[None, self.args.patch_size, self.args.patch_size, self.args.patch_size, 1])  # full surface, GT

        stn_inputs = tf.concat(values=[self.template_surface, self.target_surface], axis=4)  # input of STN, containing semi-surface from template and new sample
        self.stn = STN(stn_inputs, self.args.n_filter, self.args.l2)

        self.theta = self.stn.create_model()
        self.theta = tf.clip_by_value(self.theta, clip_value_min=0.8, clip_value_max=1.2)
        theta_mask = [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]]
        self.theta = self.theta * theta_mask

        self.template_surface_prime = spatial_transformer_network_3d(self.template_surface, self.theta)
        self.template_source_prime = spatial_transformer_network_3d(self.template_source, self.theta)

        # vnet
        vnet_input = tf.concat(values=[self.target_surface, self.template_source_prime], axis=4)
        vnet_input.set_shape([None, self.args.patch_size, self.args.patch_size, self.args.patch_size, 2])
        self.vnet = VNet(self.args.n_filter, vnet_input, self.drop, self.args.l2)
        self.target_source_prime = self.vnet.create_model()

        # loss function
        self.loss_stn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.template_surface_prime, labels=self.target_surface))
        self.loss_vnet = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.target_source_prime, labels=self.target_source))
        self.loss_all = tf.reduce_mean(0.1 * self.loss_stn + self.loss_vnet)

        # optimizer
        self.train_reg_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss_stn)
        self.train_recon_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss_vnet)
        self.train_all_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss_all)
        init_op = tf.global_variables_initializer()
        self.init_op = init_op
        count_params()
        self.sess = sess

        self.saver = tf.train.Saver(max_to_keep=25)

        exp_path = os.path.join(self.args.base_path, self.args.exp, f"cv{self.args.cv}_k{self.args.n_cluster}_size{self.args.patch_size}_{self.args.num_samples}")
        if not os.path.isdir(exp_path):
            os.makedirs(exp_path)
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
        mapping_file_path = os.path.join(args.base_path, args.data_path, f"{args.data_prefix}_{args.patch_size}_{args.num_samples}", "features", f"mapping_{args.n_cluster}.csv")
        patients = self.get_list_of_patients(data_path)
        train_patients, val_patients = get_split_deterministic(patients, fold=args.cv, num_splits=args.cv_max)
        self.train_patients = train_patients
        self.val_patients = val_patients
        dataloader_train = ReconDataloader(os.path.join(data_path, "data"), mapping_file_path, train_patients,
                                           args.batch_size, [args.patch_size, args.patch_size, args.patch_size],
                                           args.n_workers)
        dataloader_val = ReconDataloader(os.path.join(data_path, "data"), mapping_file_path, val_patients,
                                         args.batch_size, [args.patch_size, args.patch_size, args.patch_size],
                                         args.n_workers)

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
        self.mapping_df = pd.read_csv(mapping_file_path)

    def train(self, start_epoch=0):
        self.sess.run(self.init_op)
        num_train_data = len(self.train_patients)

        for epoch in tqdm(range(start_epoch, self.args.num_epochs)):
            print("[x] epoch: %d, training" % epoch)
            num_batch = num_train_data // self.args.batch_size
            epoch_loss = 0.

            for mini_batch in tqdm(range(num_batch), desc=f"training @ epoch {epoch}", unit="batch"):
                batch_data = next(self.tr_gen)

                # starting epochs
                if epoch < int(self.args.num_epochs * 0.2):
                    template_surface = np.transpose(batch_data['template_surface'], (0, 2, 3, 4, 1))
                    target_surface = np.transpose(batch_data['target_surface'], (0, 2, 3, 4, 1))

                    _, train_loss = self.sess.run([self.train_reg_op, self.loss_stn],
                                      feed_dict={self.template_surface: template_surface,
                                                 self.target_surface: target_surface,
                                                 self.lr: self.args.lr,
                                                 self.drop: self.args.dropout_p})
                # elif epoch >= int(self.args.num_epochs) * 0.2 and epoch < int(self.args.num_epochs) * 0.4:
                else:
                    template_surface = np.transpose(batch_data['template_surface'], (0, 2, 3, 4, 1))
                    target_surface = np.transpose(batch_data['target_surface'], (0, 2, 3, 4, 1))
                    template_source = np.transpose(batch_data['template_source'], (0, 2, 3, 4, 1))
                    target_source = np.transpose(batch_data['target_source'], (0, 2, 3, 4, 1))

                    _, train_loss = self.sess.run([self.train_recon_op, self.loss_vnet],
                                                   feed_dict={self.template_surface: template_surface,
                                                              self.target_surface: target_surface,
                                                              self.template_source: template_source,
                                                              self.target_source: target_source,
                                                              self.lr: self.args.lr,
                                                              self.drop: self.args.dropout_p})
                # else:
                #     template_surface = np.transpose(batch_data['template_surface'], (0, 2, 3, 4, 1))
                #     target_surface = np.transpose(batch_data['target_surface'], (0, 2, 3, 4, 1))
                #     template_source = np.transpose(batch_data['template_source'], (0, 2, 3, 4, 1))
                #     target_source = np.transpose(batch_data['target_source'], (0, 2, 3, 4, 1))

                #     _, train_loss = self.sess.run([self.train_all_op, self.loss_all],
                #                     feed_dict={self.template_surface: template_surface,
                #                                self.target_surface: target_surface,
                #                                self.template_source: template_source,
                #                                self.target_source: target_source,
                #                                self.lr: self.args.lr,
                #                                self.drop: self.args.dropout_p})
                epoch_loss += train_loss

            print("[x] epoch: %d, average loss: %f" % (epoch, epoch_loss / num_batch))

            # test model performance
            if (epoch) % self.args.validate_epoch == 0:
                print("[x] epoch: %d, validate" % epoch)
                self.test(epoch)

    def test(self, epoch):
        if not os.path.isdir(self.exp_path + "/%04d" % (epoch)):
            os.makedirs(self.exp_path + "/%04d" % (epoch))

        target = open(self.exp_path + "/%04d/val.csv" % (epoch), "w")
        target.write('patient_id,dice,le\n')

        num_val_data = len(self.val_patients)

        for mini_batch in tqdm(range(num_val_data)):
            patient_id = self.val_patients[mini_batch]
            print(f"[x] testing {patient_id}")
            template_id = self.mapping_df[self.mapping_df['sample_id'] == int(patient_id)]['cluster_id'].values[0]
            target_surface = np.load("{}/data/{}_x2.npy".format(self.data_path, patient_id), mmap_mode='r')
            target_source = np.load("{}/data/{}_y.npy".format(self.data_path, patient_id), mmap_mode='r')

            template_surface = np.load("{}/data/{}_x2.npy".format(self.data_path, template_id), mmap_mode='r')
            template_source = np.load("{}/data/{}_y.npy".format(self.data_path, template_id), mmap_mode='r')

            target_surface = np.expand_dims(np.expand_dims(target_surface, axis=3), axis=0).copy()
            template_surface = np.expand_dims(np.expand_dims(template_surface, axis=3), axis=0).copy()
            template_source = np.expand_dims(np.expand_dims(template_source, axis=3), axis=0).copy()

            target_source_pred, template_source_transformed = sess.run(
                [self.target_source_prime, self.template_source_prime],
                feed_dict={self.template_surface: template_surface,
                           self.target_surface: target_surface,
                           self.template_source: template_source,
                           self.lr: self.args.lr,
                           self.drop: self.args.dropout_p})
            target_source_pred = np.squeeze(target_source_pred)

            model_result = self.evaluate_result(np.squeeze(target_source), target_source_pred)

            dv = DataVisualizer([np.squeeze(target_surface),
                                 np.squeeze(template_surface),
                                 np.squeeze(target_source),  # GT
                                 np.squeeze(template_source),
                                 np.squeeze(target_source_pred),  # PRED
                                 np.squeeze(model_result['y_bin']),  # BIN
                                 np.squeeze(template_source_transformed)],
                                save_path=self.exp_path + "/%04d/%s.png" % (epoch, patient_id))
            dv.visualize_np(target_source_pred.shape[2], target_source_pred.shape[1], target_source_pred.shape[0])

            # post processing
            bin, dice_coeff, le = post_processing_results(np.squeeze(target_source_pred),
                                                          np.squeeze(target_source),
                                                          np.squeeze(target_surface))
            if dice_coeff == 0.:
                bin, dice_coeff, le = post_processing_results(np.squeeze(target_source_pred), np.squeeze(target_source), np.squeeze(target_surface), radius=3)
            
            bin = errosion(bin)
            dice_coeff, le = post_eval(bin, np.squeeze(target_source))

            dv2 = DataVisualizer(
                [np.squeeze(target_surface), np.squeeze(target_source_pred), bin, np.squeeze(target_source)],
                save_path=self.exp_path + "/%04d/%s_post.png" % (epoch, patient_id))
            dv2.visualize_np(target_source_pred.shape[2], target_source_pred.shape[1], target_source_pred.shape[0])

            np.save(file=self.exp_path + "/%04d/%s_target_surface.npy" % (epoch, patient_id), arr=target_surface)
            np.save(file=self.exp_path + "/%04d/%s_target_source_predicted.npy" % (epoch, patient_id), arr=target_source_pred)
            np.save(file=self.exp_path + "/%04d/%s_target_source_gt.npy" % (epoch, patient_id), arr=target_source)
            np.save(file=self.exp_path + "/%04d/%s_target_source_bin.npy" % (epoch, patient_id), arr=bin)
            np.save(file=self.exp_path + "/%04d/%s_template_source_transformed.npy" % (epoch, patient_id), arr=np.squeeze(template_source_transformed))
            np.save(file=self.exp_path + "/%04d/%s_template_source.npy" % (epoch, patient_id), arr=np.squeeze(template_source))

            target.write("%s,%f,%f\n" % (patient_id, dice_coeff, le))

        target.flush()
        target.close()

        df = pd.read_csv(self.exp_path + "/%04d/val.csv" % (epoch))
        print(f"[x] testing @ epoch {epoch}, dice = {df['dice'].mean()}, le = {df['le'].mean()}")

        self.saver.save(self.sess, os.path.join(self.exp_path, '%04d' % (epoch), 'model.cpkt'))

    def evaluate_result(self, gt, y_pred):
        y_bin, dice_coeff = evaluate_single_image(y_pred, gt)
        result = {'dice': dice_coeff, 'y_bin': y_bin}
        return result

    def load_model(self, model_path):
        print("load model from {}".format(model_path))
        self.saver.restore(self.sess, model_path)

    def predict(self, target_surface, template_surface, template_source):
        assert len(target_surface.shape) == 3
        assert len(template_surface.shape) == 3
        assert len(template_source.shape) == 3

        target_surface = np.expand_dims(np.expand_dims(target_surface, axis=3), axis=0).copy()
        template_surface = np.expand_dims(np.expand_dims(template_surface, axis=3), axis=0).copy()
        template_source = np.expand_dims(np.expand_dims(template_source, axis=3), axis=0).copy()

        target_source_pred, template_source_transformed = self.sess.run([self.target_source_prime, self.template_source_prime],
                                feed_dict={self.template_surface: template_surface,
                                           self.target_surface: target_surface,
                                           self.template_source: template_source,
                                           self.lr: self.args.lr,
                                           self.drop: self.args.dropout_p})
        target_source_pred = np.squeeze(target_source_pred)
        return target_source_pred

    def add_noise(self, x, mean=0, std=1):
        mask = np.zeros_like(x)
        mask[x>0] = 1
        noise = np.random.normal(mean, std, x.shape)
        masked_noise = noise * mask * (np.max(x) - np.min(x))
        noise_array = x + masked_noise
        return noise_array

    def test_noise(self, epoch:str, gamma):
        if not os.path.isdir(f"{self.exp_path}/{epoch}"):
            os.makedirs(f"{self.exp_path}/{epoch}")

        target = open(f"{self.exp_path}/{epoch}/val.csv", "w")
        target.write('patient_id,dice,le\n')

        num_val_data = len(self.val_patients)

        for mini_batch in tqdm(range(num_val_data)):
            patient_id = self.val_patients[mini_batch]
            template_id = self.mapping_df[self.mapping_df['sample_id'] == int(patient_id)]['cluster_id'].values[0]
            target_surface = np.load("{}/data/{}_x2.npy".format(self.data_path, patient_id), mmap_mode='r')
            target_source = np.load("{}/data/{}_y.npy".format(self.data_path, patient_id), mmap_mode='r')
            
            target_surface = self.add_noise(target_surface, mean=0, std=gamma)

            template_surface = np.load("{}/data/{}_x2.npy".format(self.data_path, template_id), mmap_mode='r')
            template_source = np.load("{}/data/{}_y.npy".format(self.data_path, template_id), mmap_mode='r')

            target_surface = np.expand_dims(np.expand_dims(target_surface, axis=3), axis=0).copy()
            template_surface = np.expand_dims(np.expand_dims(template_surface, axis=3), axis=0).copy()
            template_source = np.expand_dims(np.expand_dims(template_source, axis=3), axis=0).copy()

            target_source_pred, template_source_transformed = sess.run(
                [self.target_source_prime, self.template_source_prime],
                feed_dict={self.template_surface: template_surface,
                           self.target_surface: target_surface,
                           self.template_source: template_source,
                           self.lr: self.args.lr,
                           self.drop: self.args.dropout_p})
            target_source_pred = np.squeeze(target_source_pred)

            # post processing
            bin, dice_coeff, le = post_processing_results(np.squeeze(target_source_pred),
                                                          np.squeeze(target_source),
                                                          np.squeeze(target_surface), radius=6)
            if dice_coeff == 0.:
                bin, dice_coeff, le = post_processing_results(np.squeeze(target_source_pred), np.squeeze(target_source), np.squeeze(target_surface), radius=3)

            bin = errosion(bin)
            dice_coeff, le = post_eval(bin, np.squeeze(target_source))

            dv2 = DataVisualizer(
                [np.squeeze(target_surface), np.squeeze(target_source_pred), bin, np.squeeze(target_source)],
                save_path=f"{self.exp_path}/{epoch}/{patient_id}_post.png")
            dv2.visualize_np(target_source_pred.shape[2], target_source_pred.shape[1], target_source_pred.shape[0])

            np.save(file=f"{self.exp_path}/{epoch}/{patient_id}_target_surface.npy", arr=target_surface)
            np.save(file=f"{self.exp_path}/{epoch}/{patient_id}_target_source_predicted.npy", arr=target_source_pred)
            np.save(file=f"{self.exp_path}/{epoch}/{patient_id}_target_source_gt.npy", arr=target_source)
            np.save(file=f"{self.exp_path}/{epoch}/{patient_id}_target_source_bin.npy", arr=bin)
            np.save(file=f"{self.exp_path}/{epoch}/{patient_id}_template_source_transformed.npy", arr=np.squeeze(template_source_transformed))
            np.save(file=f"{self.exp_path}/{epoch}/{patient_id}_template_source.npy", arr=np.squeeze(template_source))

            target.write("%s,%f,%f\n" % (patient_id, dice_coeff, le))

        target.flush()
        target.close()

        df = pd.read_csv(f"{self.exp_path}/{epoch}/val.csv")
        print(f"[x] testing noise @ epoch {epoch}, gamma = {gamma}, dice = {df['dice'].mean()}, le = {df['le'].mean()}")


    def test_dual(self, data_path_root):
        eval_df = pd.DataFrame(columns=["epoch", "subject_id", "thresh", "dice", "cluster_i", "cluster_j", "le1", "le2"])

        data_save_path = f"{self.exp_path}/double_source_pred"
        if not os.path.isdir(data_save_path):
            os.makedirs(data_save_path)
        
        mapping_file_path = os.path.join(self.args.base_path, self.args.data_path, f"{self.args.data_prefix}_{self.args.patch_size}_{self.args.num_samples}", "features", f"mapping_{self.args.n_cluster}.csv")
        mapping_df = pd.read_csv(mapping_file_path)

        template_ids = np.unique(mapping_df["cluster_id"].values)
        data_template_surfaces, data_template_sources = [], []
        for template_id in template_ids:
            data_template_surface = np.load(f"{self.args.base_path}/{self.args.data_path}/{self.args.data_prefix}_{self.args.patch_size}_{self.args.num_samples}/data/{template_id}_x2.npy")
            data_template_source = np.load(f"{self.args.base_path}/{self.args.data_path}/{self.args.data_prefix}_{self.args.patch_size}_{self.args.num_samples}/data/{template_id}_y.npy")
            data_template_surfaces.append(data_template_surface)
            data_template_sources.append(data_template_source)

        for subject in tqdm(range(1, 101)):
            data_path = os.path.join(data_path_root, f"{subject}.mat")
            mat = scio.loadmat(data_path)
            data_full_surfaces = []
            full_surface = np.pad(mat[f'output_{1}'], [(2, 2), (6, 6), (4, 4)], mode='constant', constant_values=0)
            data_full_surfaces.append(full_surface)
            full_surface = np.pad(mat[f'output_{2}'], [(2, 2), (6, 6), (4, 4)], mode='constant', constant_values=0)
            data_full_surfaces.append(full_surface)
            
            target_source_gt = np.pad(mat['target'], [(2, 2), (6, 6), (4, 4)], mode='constant', constant_values=0)

            global_dice, global_thresh, global_cluster = 0., 0., (0, 0)
            for thresh in tqdm(np.arange(0.1, 0.31, 0.1)):
                dice, (cla_i, cla_j), results = self.predict3(data_template_surfaces, data_template_sources,
                                                         data_full_surfaces, target_source_gt,
                                                         subject, data_save_path, self.args.n_cluster, thresh, mapping_df)
                if dice>global_dice:
                    global_dice = dice
                    global_thresh = thresh
                    # data_save_path, f"{subject_id}_{global_class[0]}_{global_class[1]}.png"
                    global_cluster = (cla_i, cla_j)
                    np.save(f"{data_save_path}/{subject}_1.npy", results[0])
                    np.save(f"{data_save_path}/{subject}_2.npy", results[1])
                    np.save(f"{data_save_path}/{subject}_source.npy", target_source_gt)
                    np.save(f"{data_save_path}/{subject}_surface_1.npy", data_full_surfaces[0])
                    np.save(f"{data_save_path}/{subject}_surface_2.npy", data_full_surfaces[1])
                    
            eval_df.loc[eval_df.shape[0]] = {"subject_id": subject, 
                                             "thresh": global_thresh, "dice": global_dice,
                                             "cluster_i": global_cluster[0], "cluster_j": global_cluster[1]}
        
        eval_df.to_csv(f"{data_save_path}/dual.csv", index=False)
        print(f"[x] test_dual, dice = {eval_df['dice'].mean()}, le1 = {eval_df['le1'].mean()}, le2 = {eval_df['le2'].mean()}")

    def predict3(self, data_template_surfaces, data_template_sources, data_full_surfaces, target_source_gt, subject_id, 
                    data_save_path, n_cluster, thresh, mapping_df):
        global_dice, global_class = 0.0, (0, 0)

        results = None
        for clazz_i in range(n_cluster):
            for clazz_j in range(n_cluster):
                can_generate_mask = True
                semi_surfaces = []
                target_source_preds = []
                target_source_bins = []
                target_source_gts = []
                
                try:
                    for i in range(1, 3):
                        full_surface = np.copy(data_full_surfaces[i-1])
                        T = (np.max(full_surface[np.nonzero(full_surface)]) - np.min(full_surface[np.nonzero(full_surface)])) * thresh
                        semi_surface = full_surface.copy()
                        semi_surface[semi_surface < T] = 0

                        if i ==1:
                            template_id = mapping_df[mapping_df["pseudo_label"] == clazz_i]
                        else:
                            template_id = mapping_df[mapping_df["pseudo_label"] == clazz_j]
                        # template_id = np.unique(template_id['cluster_id'])[0]
                        data_template_surface = data_template_surfaces[clazz_i]
                        data_template_source = data_template_sources[clazz_j]
                        target_source_pred = self.predict(semi_surface, data_template_surface, data_template_source)
                        target_source_bin = post_processing_results_dual(target_source_pred, semi_surface, radius=6)
                        target_source_bin = errosion(target_source_bin)

                        semi_surfaces.append(semi_surface)
                        target_source_preds.append(target_source_pred)
                        target_source_bins.append(target_source_bin)
                        target_source_gts.append(target_source_gt)
                except:
                    can_generate_mask = False

                if can_generate_mask:
                    target_source_bin_fusion = np.zeros_like(target_source_bins[0])
                    target_source_bin_fusion[target_source_bins[0] > 0] = 1
                    target_source_bin_fusion[target_source_bins[1] > 0] = 1

                    dice_coeff = dice_coefficient_in_train(target_source_bin_fusion, target_source_gts[0])

                    if dice_coeff > global_dice:
                        global_dice = dice_coeff
                        global_class = (clazz_i, clazz_j)
                        print(f"subject_id = {subject_id}, dice = {global_dice}, clazz_i = {clazz_i}, clazz_j = {clazz_j}")

                        dv = DataVisualizer([semi_surfaces[0],
                                             semi_surfaces[1],
                                             target_source_preds[0],
                                             target_source_preds[1],
                                             target_source_bins[0],
                                             target_source_bins[1],
                                             target_source_gts[0],
                                             target_source_bin_fusion], 
                                             save_path=os.path.join(data_save_path, f"{subject_id}_{global_class[0]}_{global_class[1]}.png"))
                        dv.visualize_np(64, 64, 64, 1)
                        results = target_source_bins
                        # imageio.imsave(os.path.join(data_save_path, f"{subject_id}_{global_class[0]}_{global_class[1]}.png"), img)

        return global_dice, global_class, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--base_path', type=str, default="/media/z/data3/wei")
    parser.add_argument('--exp', type=str, default="exp_single")
    parser.add_argument('--num_epochs', type=int, default=201)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--validate_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)

    # data
    # cpu
    parser.add_argument('--n_workers', type=int, default=6)

    # gpu
    parser.add_argument('--gpu', type=str, default="0")

    # CV
    parser.add_argument('--cv', type=int, default=0)  # cross validation, CV=5
    parser.add_argument('--cv_max', type=int, default=5)

    # Model
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--data_path', type=str, default="data_prepare/MCX")
    parser.add_argument('--data_prefix', type=str, default="shape_single")
    parser.add_argument('--n_filter', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--dropout_p', type=float, default=0.8)
    parser.add_argument('--l2', type=float, default=0.2)
    parser.add_argument('--n_cluster', type=int, default=20)


    parser.add_argument('--train', type=str, default="dual")

    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--model_path', type=str, default="/media/z/data3/wei/exp_single/cv0_k20_size64_1000/0160/model.cpkt")
    parser.add_argument('--dual_data_path', type=str, default="/media/z/data3/wei/data_prepare/blind_source_separation/speard")

    params = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(params.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    trainer = ReconBinTrainer3D(params, sess)
    if params.train == "train":
        exp_path = os.path.join(params.base_path, params.exp, f"cv{params.cv}_k{params.n_cluster}_size{params.patch_size}_{params.num_samples}")
        if os.path.isdir(exp_path):
            start_epoch = 0
            has_epoch_dir = False
            for d in sorted(os.listdir(exp_path)):
                if os.path.isdir(f"{exp_path}/{d}"):
                    has_epoch_dir = True
                    if int(d) > start_epoch:
                        start_epoch = int(d)
            if has_epoch_dir:
                model_path = f"{exp_path}/{start_epoch:04d}/model.cpkt"
                print(f"[x] load model from {model_path}")
                trainer.load_model(model_path)
                trainer.train(start_epoch)
            else:
                trainer.train(start_epoch)
        else: 
            trainer.train(0)
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
    elif params.train == "dual":
        trainer.load_model(params.model_path)
        trainer.test_dual(params.dual_data_path)