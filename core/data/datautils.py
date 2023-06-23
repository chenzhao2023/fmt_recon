import os
import numpy as np
from glob import glob
from sklearn.model_selection import KFold


def get_list_of_patients(data_folder):
    patients = []
    full_surface_npy_files = glob(os.path.join(data_folder, "*x.npy"))
    for full_surface_npy_file in full_surface_npy_files:
        patient_name = full_surface_npy_file[full_surface_npy_file.rfind("/")+1: full_surface_npy_file.rfind("_full_surface")]
        patients.append(patient_name)
    return patients


def get_split_deterministic(all_keys, fold=0, num_splits=5, random_state=12345):
    """
    Splits a list of patient identifiers (or numbers) into num_splits folds and returns the split for fold fold.
    :param all_keys:
    :param fold:
    :param num_splits:
    :param random_state:
    :return:
    """
    all_keys_sorted = np.sort(list(all_keys))
    splits = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(splits.split(all_keys_sorted)):
        if i == fold:
            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]
            break
    return train_keys, test_keys


if __name__ == '__main__':
    root_path = "/home/z/Desktop/style_transfer/data"
    print(get_list_of_patients(root_path))