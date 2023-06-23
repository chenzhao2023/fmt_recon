import numpy as np
import pandas as pd
from batchgenerators.dataloading.data_loader import DataLoader


class ReconDataloader(DataLoader):

    def __init__(self,
                 data_path,
                 mapping_file_path,
                 data,
                 batch_size,
                 patch_size,
                 num_threads_in_multithreaded,
                 seed_for_shuffle=1234,
                 return_incomplete=False,
                 shuffle=True,
                 infinite=True):

        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle, infinite)

        self.data_path = data_path
        self.patch_size = patch_size
        self.input_channel = 1
        self.output_channel = 1
        self.indices = list(range(len(data)))
        self.mapping_df = pd.read_csv(mapping_file_path)

    def load_patient(self, patient_id):
        template_id = self.mapping_df[self.mapping_df['sample_id']==int(patient_id)]['cluster_id'].values[0]

        target_surface   = np.load("{}/{}_x2.npy".format(self.data_path, patient_id), mmap_mode='r')
        target_source   = np.load("{}/{}_y.npy".format(self.data_path, patient_id), mmap_mode='r')

        template_surface = np.load("{}/{}_x2.npy".format(self.data_path, template_id), mmap_mode='r')
        template_source = np.load("{}/{}_y.npy".format(self.data_path, template_id), mmap_mode='r')

        # semi_surface[  semi_surface   > 0] = 1
        # full_surface[  full_surface   > 0] = 1
        # semi_surface_t[semi_surface_t > 0] = 1
        # full_surface_t[full_surface_t > 0] = 1

        target_surface = np.expand_dims(target_surface, axis=0)
        target_source = np.expand_dims(target_source, axis=0)

        template_surface = np.expand_dims(template_surface, axis=0)
        template_source = np.expand_dims(template_source, axis=0)

        return target_surface, target_source, template_surface, template_source

    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for data and seg
        target_surfaces = np.zeros((self.batch_size, self.output_channel, *self.patch_size), dtype=np.float32)
        target_sources = np.zeros((self.batch_size, self.input_channel, *self.patch_size), dtype=np.float32)
        template_surfaces = np.zeros((self.batch_size, self.output_channel, *self.patch_size), dtype=np.float32)
        template_sources = np.zeros((self.batch_size, self.input_channel, *self.patch_size), dtype=np.float32)

        patient_names = []

        for i, j in enumerate(patients_for_batch):
            target_surface, target_source, template_surface, template_source = self.load_patient(j)

            target_surfaces[i] = target_surface
            target_sources[i] = target_source
            template_surfaces[i] = template_surface
            template_sources[i] = template_source

            patient_names.append(j)

        # data = np.transpose(data, (0, 2, 3, 4, 1))
        # seg = np.transpose(seg, (0, 2, 3, 4, 1))

        return {'target_surface': target_surfaces, 'target_source': target_sources, 'template_surface': template_surfaces, 'template_source': template_sources}


class ReconDataloaderVNet(DataLoader):

    def __init__(self,
                 data_path,
                 data,
                 batch_size,
                 patch_size,
                 num_threads_in_multithreaded,
                 seed_for_shuffle=1234,
                 return_incomplete=False,
                 shuffle=True,
                 infinite=True):

        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle, infinite)

        self.data_path = data_path
        self.patch_size = patch_size
        self.input_channel = 1
        self.output_channel = 1
        self.indices = list(range(len(data)))

    def load_patient(self, patient_id):
        target_surface   = np.load("{}/{}_x2.npy".format(self.data_path, patient_id), mmap_mode='r')
        target_source   = np.load("{}/{}_y.npy".format(self.data_path, patient_id), mmap_mode='r')

        target_surface = np.expand_dims(target_surface, axis=0)
        target_source = np.expand_dims(target_source, axis=0)

        return target_surface, target_source

    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for data and seg
        target_surfaces = np.zeros((self.batch_size, self.output_channel, *self.patch_size), dtype=np.float32)
        target_sources = np.zeros((self.batch_size, self.input_channel, *self.patch_size), dtype=np.float32)

        patient_names = []

        for i, j in enumerate(patients_for_batch):
            target_surface, target_source = self.load_patient(j)

            target_surfaces[i] = target_surface
            target_sources[i] = target_source

            patient_names.append(j)


        return {'target_surface': target_surfaces, 'target_source': target_sources}

