import numpy as np
import os
import matplotlib.pyplot as plt

from radiomics import featureextractor, getTestCase
from glob import glob
from tqdm import tqdm


from core.utils.processing_data import generate_vex_files


def get_list_of_patients(data_folder):
    patients = []
    full_surface_npy_files = glob(os.path.join(data_folder, "*x.npy"))
    for full_surface_npy_file in full_surface_npy_files:
        patient_name = full_surface_npy_file[full_surface_npy_file.rfind("/")+1: full_surface_npy_file.rfind("_x")]
        patients.append(patient_name)
    return patients


if __name__ == '__main__':
    root_path = "/media/z/data3/wei/data_prepare"
    shape_size = 64
    num_samples = 1000
    data_path = f"{root_path}/MCX/shape_single_{shape_size}_{num_samples}/data"
    
    feature_save_path = f"{root_path}/MCX/shape_single_{shape_size}_{num_samples}/features"
    
    patients = sorted(get_list_of_patients(os.path.join(root_path, data_path)))

    if not os.path.isdir(os.path.join(root_path, feature_save_path)):
        os.makedirs(os.path.join(root_path, feature_save_path))

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('ngtdm')

    ns = []
    for patient_id in tqdm(patients):
        data_x = np.load("{}/{}_x2.npy".format(data_path, patient_id), mmap_mode='r')
        n = generate_vex_files(data_x, extractor)

        ns.append(n)

    features = np.array(ns)
    patients = np.array(patients)
    np.save(file=os.path.join(root_path, feature_save_path, "features.npy"), arr=features)
    np.save(file=os.path.join(root_path, feature_save_path, "patients.npy"), arr=patients)
    # print(features.shape)


