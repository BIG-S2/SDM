"""
Run SDM (statistical disease mapping) pipeline on ADNI hippocampal surface data (step 2)
Usage: python ./sdm_hippo_s2.py ./data/ ./result/left/ left 10 5

Author: Rongjie Liu (rongjie.liu@rice.edu)
Last update: 2019-2-28
"""

import sys
import os
import numpy as np
from numpy.linalg import inv, eigh
from mrf import mrf_map
# from sklearn.cluster import KMeans
from scipy.io import loadmat
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image

"""
installed all the libraries above
"""

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    phase = sys.argv[3]
    em_iter = int(sys.argv[4])
    map_iter = int(sys.argv[5])

    if not os.path.exists(output_dir + phase):
        os.mkdir(output_dir + phase)

    """+++++++++++++++++++++++++++++++++++"""
    print("""\n Load dataset & preprocessing \n """)
    # find the index for functional data downsampling #
    idd = np.arange(15000)
    idd_mat = idd.reshape(150, 100)
    rs_idd_mat = idd_mat[::2, ::2]
    rs_idd = rs_idd_mat.reshape(rs_idd_mat.shape[0] * rs_idd_mat.shape[1], )
    print("+++++++Read the mask file+++++++")
    mask = np.ones(shape=(150, 100))
    coord_tmp = np.transpose(np.nonzero(mask))
    coord_mat = coord_tmp[rs_idd, :]
    n_v, d = coord_mat.shape
    print("The matrix dimension of coordinate data is " + str(coord_mat.shape))
    y_file_name = input_dir + "y_" + phase + ".mat"
    mat = loadmat(y_file_name)
    y_mat = mat['y_' + phase]
    if len(y_mat.shape) == 2:
        y_mat = y_mat.reshape(y_mat.shape[0], y_mat.shape[1], 1)
    y_data = y_mat[:, rs_idd, 0:2]
    n, l, m = y_data.shape
    print("The matrix dimension of imaging data is " + str(y_data.shape))
    print("+++++++Read the covariate data+++++++")
    x_file_name = input_dir + "x.txt"
    x = np.loadtxt(x_file_name)
    x[:, 1] = (x[:, 1]-np.mean(x[:, 1]))/np.std(x[:, 1])
    x_data = np.hstack((np.ones(shape=(n, 1)), x))
    p = x_data.shape[1]
    print("The design matrix dimension is " + str(x_data.shape))
    print("+++++++Read the label data+++++++")
    z_file_name = input_dir + "dx.txt"
    z = np.loadtxt(z_file_name)
    print("The sample sizes for normal controls and patients are " + str(n-sum(z)) + "and" + str(sum(z)))

    """++++++++++++++++++++++++++++++++++++"""
    print("""\n Loading results from step 1 \n """)
    beta_file_name = output_dir + "beta.mat"
    mat = loadmat(beta_file_name)
    b_0 = mat['beta']
    sigma_file_name = output_dir + "sigma.mat"
    mat = loadmat(sigma_file_name)
    sigma_eta = mat['sigma_eta']
    omega_eps = mat['omega_eps']
    res_y = y_data * 0
    for j in range(m):
        res_y[:, :, j] = y_data[:, :, j] - np.dot(x_data, b_0[:, :, j])
    inv_s = np.zeros(shape=(n_v, m, m))
    for l in range(n_v):
        if m > 1:
            inv_s2 = inv(sigma_eta[l, :, :]+omega_eps[l, :, :])
            w, v = eigh(np.squeeze(inv_s2))
            w = np.real(w)
            w[w < 0] = 0
            w_diag = np.diag(w ** (1 / 2))
            inv_s_tp = np.dot(np.dot(v, w_diag), v.T)
            inv_s[l, :, :] = np.real(inv_s_tp)
        else:
            inv_s[l, :, :] = (sigma_eta[l, :, :]+omega_eps[l, :, :])**(-0.5)

    """++++++++++++++++++++++++++++++++++++"""
    print("""\n Threshold on the difference of images to get initial disease regions and effects\n""")
    y1 = y_data[np.nonzero(z == 1)[0], :, :]
    x1 = x_data[np.nonzero(z == 1)[0], :]
    n1 = int(sum(z))
    res_y1 = res_y[np.nonzero(z == 1)[0], :, :]
    # threshold = np.percentile(np.squeeze(res_y1[:, :, 0]), 5, axis=1)
    y1_dif_p = 0 * res_y1
    label0 = np.zeros(shape=(n1, n_v))
    for i in range(n1):
        res_img = res_y1[i, :, 0].reshape(75, 50)
        graph = image.img_to_graph(res_img)
        clustering_labels = spectral_clustering(graph, n_clusters=5, eigen_solver='arpack')
        labels_id = np.unique(clustering_labels)
        id_size = len(labels_id)
        mu = np.zeros(shape=id_size)
        for tt in range(id_size):
            mu[tt] = np.mean(res_y1[i, clustering_labels == labels_id[tt], 0])
        idx_mu = np.argsort(mu)
        idx_label0 = np.nonzero(clustering_labels == labels_id[idx_mu[0]])[0]
        label0[i, idx_label0] = 1
    label0_sum = np.sum(label0, axis=0)
    idx_sum_label0 = np.nonzero(label0_sum > 50)[0]
    inv_s_0 = np.sum(inv_s[idx_sum_label0, :, :], axis=0)
    bbar_0 = np.zeros(shape=(p, m))
    for k in range(len(idx_sum_label0)):
        sub_idx = np.nonzero(label0[:, idx_sum_label0[k]] == 1)[0]
        y1_dif_p = np.dot(np.squeeze(res_y1[sub_idx, idx_sum_label0[k], :]),
                          np.squeeze(inv_s[idx_sum_label0[k], :, :]))
        bbar_0 = bbar_0 + np.dot(np.dot(inv(np.dot(x1[sub_idx, :].T, x1[sub_idx, :])), x1[sub_idx, :].T),
                                 y1_dif_p)
    bbar_0 = np.dot(bbar_0, inv(inv_s_0))
    print(bbar_0)
    mu = np.dot(x1, bbar_0)

    """++++++++++++++++++++++++++++++++++++"""
    print("""\n Run EM algorithm\n""")
    gamma = 0.2
    mask_new = np.ones(shape=(75, 50))
    coord_mat_new = np.transpose(np.nonzero(mask_new))
    for k in range(em_iter):
        print("iteration %d \n" % (k+1))
        label0 = mrf_map(label0, res_y1, mask_new, coord_mat_new, mu, inv_s, gamma, map_iter)
        label0_sum = np.sum(label0, axis=0)
        idx_sum_label0 = np.nonzero(label0_sum > 50)[0]
        inv_s_0 = np.sum(inv_s[idx_sum_label0, :, :], axis=0)
        bbar_0 = np.zeros(shape=(p, m))
        for l in range(len(idx_sum_label0)):
            sub_idx = np.nonzero(label0[:, idx_sum_label0[k]] == 1)[0]
            y1_dif_p = np.dot(np.squeeze(res_y1[sub_idx, idx_sum_label0[l], :]),
                              np.squeeze(inv_s[idx_sum_label0[k], :, :]))
            bbar_0 = bbar_0 + np.dot(np.dot(inv(np.dot(x1[sub_idx, :].T, x1[sub_idx, :])), x1[sub_idx, :].T),
                                     y1_dif_p)
        bbar_0 = np.dot(bbar_0, inv(inv_s_0))
        print(bbar_0)
        mu = np.dot(x1, bbar_0)
    """++++++++++++++++++++++++++++++++++++"""
    print("""\n Save results\n """)
    label_file_name = output_dir + "label.txt"
    np.savetxt(label_file_name, label0)
    bbar_file_name = output_dir + "bbar.txt"
    np.savetxt(bbar_file_name, bbar_0[:, :])
    coord_file_name = output_dir + "coord.txt"
    np.savetxt(coord_file_name, coord_mat)
