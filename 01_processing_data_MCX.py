import numpy as np
import os
import re
import scipy.io as scio
import skimage.measure
from glob import glob
from core.utils.visualizer import DataVisualizer
from scipy.ndimage import zoom
from scipy import ndimage

from tqdm import tqdm


def pad(image, target_size):
    diff = np.abs(np.array(target_size, target_size, target_size) - image.shape)
    image = np.pad(image, [(0, 1), (0, 1)], mode='constant', constant_values=0)
    return image

def resample(image, new_shape_size):
    real_resize_factor = np.array([new_shape_size, new_shape_size, new_shape_size]) / image.shape
    real_resize_factor = np.round(real_resize_factor)
    #image = ndimage.interpolation.zoom(image, real_resize_factor, order=0)
    # resize_factor = np.array([new_shape_size, new_shape_size, new_shape_size]) / image.shape


    # not use
    # print(f"{np.min(image)}, {np.max(image)}")
    # image = ndimage.zoom(image, resize_factor)
    # image = (image - np.min(image))/(np.max(image)-np.min(image))
    # print(f"{np.min(image)}, {np.max(image)}")
    # return image
    image = skimage.measure.block_reduce(image, (real_resize_factor), np.max)
    image = pad(image, new_shape_size)
    return image


if __name__ == '__main__':
    # hyper parameters
    num_samples = 1000
    root_path = "/media/z/data3/wei/data_prepare"
    data_dir = "MCX/data1000"
    new_shape_size = 64
    data_save_path = f"MCX/shape_single_{new_shape_size}_{num_samples}"
    thresh = 0.2

    if not os.path.isdir(os.path.join(root_path, data_save_path)):
        os.makedirs(os.path.join(root_path, data_save_path))
        os.makedirs(os.path.join(root_path, data_save_path, "visualize"))
        os.makedirs(os.path.join(root_path, data_save_path, "data"))

    #n_samples = len(glob(os.path.join(root_path, data_dirs[0], "*.mat")))
    #print(n_samples)
    label_target = open(os.path.join(root_path, data_save_path, "label.csv"), "w")
    label_target.write("id,label\n")
    mat_files = sorted(glob(os.path.join(root_path, data_dir, "*.mat")))
    for mat_file in tqdm(sorted(mat_files)):
        mat_name = mat_file[mat_file.rfind("/")+1:mat_file.rfind(".mat")]
        if mat_name.rfind("_sign") == -1:
            if int(mat_name) <= num_samples:
                mat = scio.loadmat(mat_file)
                mat_sign_file = os.path.join(root_path, data_dir, f"{int(mat_name)}_sign.mat")
                sign_mat = scio.loadmat(mat_sign_file)
                label_target.write(f"{int(mat_name)},{np.squeeze(sign_mat['sign'])}\n") # TODO
                data_x, data_y = mat['voxel_1'], mat['voxel_3']

                data_x = np.pad(data_x, [(2, 2), (6, 6), (4, 4)], mode='constant', constant_values=0)
                data_y = np.pad(data_y, [(2, 2), (6, 6), (4, 4)], mode='constant', constant_values=0)

                T = (np.max(data_x[np.nonzero(data_x)]) - np.min(data_x[np.nonzero(data_x)])) * thresh
                data_x2 = data_x.copy()
                data_x2[data_x2<T] = 0

                data_visualize_list = [data_x, data_x2, data_y]
                data_visualizer = DataVisualizer(data_visualize_list, os.path.join(root_path, data_save_path, "visualize", f"{mat_name}.png"))
                data_visualizer.visualize_np(data_x.shape[2], patch_size_x=data_x.shape[0], patch_size_y=data_x.shape[1])
                #data_visualizer.visualize(data_x.shape[2])
                #full_surface[full_surface>0] = 1
                #semi_surface[semi_surface>0] = 1

                np.save(file=os.path.join(root_path, data_save_path, "data", f"{mat_name}_x.npy"), arr=data_x)
                np.save(file=os.path.join(root_path, data_save_path, "data", f"{mat_name}_x2.npy"), arr=data_x2)
                np.save(file=os.path.join(root_path, data_save_path, "data", f"{mat_name}_y.npy"), arr=data_y)