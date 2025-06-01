"""
Ref:
    [1]	W. Sun, L. Zhang, B. Du, W. Li, and Y. Mark Lai, "Band Selection Using Improved Sparse Subspace Clustering
    for Hyperspectral Imagery Classification," IEEE Journal of Selected Topics in Applied Earth Observations and
    Remote Sensing, vol. 8, pp. 2784-2797, 2015.

Formula:
    arg min ||X - XW||_F + lambda||W||_F subject to diag(Z) = 0
Solution:
    Wˆ = −(X^T X + lambda*I)^−1 (diag((X^T X + lambda*I)−1))^−1
"""

import numpy as np
from sklearn.cluster import SpectralClustering
import scipy.io as scio
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt
from matplotlib import cm


class ISSC_HSI(object):
    """
    :argument:
        Implementation of L2 norm based sparse self-expressive clustering model
        with affinity measurement basing on angular similarity
    """
    def __init__(self, n_band=5, coef_=1):
        self.n_band = n_band
        self.coef_ = coef_

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X):
        """
        :param X: shape [n_row*n_clm, n_band]
        :return: selected band subset
        """
        I = np.eye(X.shape[1])
        coefficient_mat = -1 * np.dot(np.linalg.inv(np.dot(X.transpose(), X) + self.coef_ * I),
                                      np.linalg.inv(np.diag(np.diag(np.dot(X.transpose(), X) + self.coef_ * I))))
        temp = np.linalg.norm(coefficient_mat, axis=0).reshape(1, -1)
        affinity = (np.dot(coefficient_mat.transpose(), coefficient_mat) /
                    np.dot(temp.transpose(), temp))**2


        # 定义三维数据
        xx = np.arange(400, 720, 10)
        yy = np.arange(400, 720, 10)
        row, col = np.diag_indices_from(affinity)
        z = affinity
        z[row, col] = 0
        X, Y = np.meshgrid(xx, yy)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        print(X.shape, Y.shape, z.shape)
        ax.plot_surface(X, Y, z, cmap=cm.gist_rainbow)
        ax.set_zlim(0, 0.1)
        plt.show()

        sc = SpectralClustering(n_clusters=self.n_band, affinity='precomputed')
        sc.fit(affinity)
        selected_band, bands_list = self.__get_band(sc.labels_, X)
        return selected_band, bands_list

    def __get_band(self, cluster_result, X):
        """
        select band according to the center of each cluster
        :param cluster_result:
        :param X:
        :return:
        """
        print(cluster_result)
        selected_band = []
        bands_list = []
        n_cluster = np.unique(cluster_result).__len__()
        # img_ = X.reshape((n_row * n_column, -1))  # n_sample * n_band
        for c in np.unique(cluster_result):
            idx = np.nonzero(cluster_result == c)
            center = np.mean(X[:, idx[0]], axis=1).reshape((-1, 1))
            distance = np.linalg.norm(X[:, idx[0]] - center, axis=0)
            band_ = X[:, idx[0]][:, distance.argmin()]
            # print(X.shape)
            # print(band_.shape)
            selected_band.append(band_)
            # print('波段集合：', idx[0])
            # print('选择波段：', idx[0][distance.argmin()])
            bands_list.append(idx[0][distance.argmin()])
        bands = np.asarray(selected_band).transpose()
        # bands = bands.reshape(n_cluster, n_row, n_column)
        # bands = np.transpose(bands, axes=(1, 2, 0))
        return bands, bands_list


if __name__ == '__main__':
    # Load 3D data cube
    dataFile1 = r'I:\sprebuilding\output\train\mat\3.mat'
    data1 = scio.loadmat(dataFile1)
    input = data1['input']
    output = data1['label']
    wavelength = np.arange(400, 701, 10)  # 波长
    print('Data loading completed')

    n_row, n_column, n_band = output.shape
    img = minmax_scale(output.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    x_input = img.reshape(n_row*n_column, n_band)
    num_class = 5
    ISSC = ISSC_HSI()
    bands, bands_list = ISSC.predict(x_input)
    bands_result = [wavelength[i] for i in bands_list]
    print('Band selection result：', bands_result)
    print(bands_list)



    #
