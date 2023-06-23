from sklearn.cluster import MiniBatchKMeans
import os
import matplotlib.pyplot as plt
import numpy as np
import joblib

from core.data.datautils import get_list_of_patients
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

if __name__ == '__main__':
    root_path = "/media/z/data3/wei/data_prepare"
    shape_size = 64
    num_samples = 1000
    # data_path = f"{root_path}/MCX/shape_{shape_size}/data" # SINGLE SOURCE
    # feature_save_path = f"{root_path}/MCX/shape_{shape_size}/features" # SINGLE SOURCE
    data_path = f"{root_path}/MCX/shape_single_{shape_size}_{num_samples}/data"  # DOUBLE SOURCE
    feature_save_path = f"{root_path}/MCX/shape_single_{shape_size}_{num_samples}/features"  # DOUBLE SOURCE

    n_cluster = 4


    # load features
    features = np.load(os.path.join(root_path, feature_save_path, "features.npy"))
    patients = np.load(os.path.join(root_path, feature_save_path, "patients.npy"))

    # Min-Max scaler
    scaler = MinMaxScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    # clustering
    mbk = MiniBatchKMeans(
        init="k-means++", n_clusters=n_cluster, batch_size=32, n_init=10, max_no_improvement=10, verbose=0)
    mbk.fit(features)
    print(mbk.cluster_centers_)



    # PCA
    pca = PCA(n_components=2)
    pca.fit(features)
    print(pca.explained_variance_ratio_)
    pca_features = pca.transform(features)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    colors = get_cmap(n_cluster, "jet")
    mbk_means_labels = pairwise_distances_argmin(features,  mbk.cluster_centers_)

    target = open(os.path.join(root_path, feature_save_path, f"mapping_{n_cluster}.csv"), "w")
    target.write("sample_id,cluster_id,pseudo_label\n")

    for k in range(n_cluster):
        my_members = mbk_means_labels == k
        ax.plot(pca_features[my_members, 0], pca_features[my_members, 1], "w", markerfacecolor=colors(k), marker=".")

    # find nearest samples
    cluster_center_sample_indexs = []
    for k in range(n_cluster):
        cluster_center = mbk.cluster_centers_[k]
        current_min = np.inf
        sample_index = 0
        my_members = mbk_means_labels == k
        for i in range(features.shape[0]):
            d = np.linalg.norm(features[i] - cluster_center)
            if d < current_min:
                current_min = d
                sample_index = i
        print(f"cluster {k}, sample_index = {sample_index}, pca coord = {pca_features[sample_index]}, # samples = {np.sum(my_members>0)}")
        cluster_center_sample_indexs.append(sample_index)

        ax.plot(
            pca_features[sample_index][0],
            pca_features[sample_index][1],
            "o",
            markerfacecolor=colors(k),
            markeredgecolor="k",
            markersize=6,
        )
        # write to file
        for i in range(features.shape[0]):
            my_members = mbk_means_labels == k
            if i in np.where(my_members==1)[0]:
                target.write(f"{patients[i]},{patients[sample_index]},{k}\n")

    ax.set_title("KMeans Results")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    #plt.show()
    plt.savefig(os.path.join(root_path, feature_save_path, f"mapping_{n_cluster}.png"))
    joblib.dump(mbk, os.path.join(root_path, feature_save_path, 'clf.joblib'))
    target.close()