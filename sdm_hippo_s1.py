"""
Run SDM (statistical disease mapping) pipeline on ADNI hippocampal surface data (step 1)
Usage: python ./sdm_hippo_s1.py ./data/ ./result/ left

Author: Rongjie Liu (rongjie.liu@rice.edus)
Last update: 2018-2-28
"""

import sys
import os
import numpy as np
from numpy.linalg import inv
from mvcm import mvcm
from scipy.io import loadmat, savemat

"""
installed all the libraries above
"""

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    phase = sys.argv[3]

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
    print("The design matrix dimension is " + str(x_data.shape))
    print("+++++++Read the label data+++++++")
    z_file_name = input_dir + "dx.txt"
    z = np.loadtxt(z_file_name)
    print("The sample sizes for normal controls and patients are " + str(n-sum(z)) + "and" + str(sum(z)))

    """++++++++++++++++++++++++++++++++++++"""
    print("""\n Run MVCM on normal controls \n """)
    y0 = y_data[np.nonzero(z == 0)[0], :, :]
    x0 = x_data[np.nonzero(z == 0)[0], :]
    """ calculate the hat matrix """
    p = x0.shape[1]
    c_mat = np.dot(inv(np.dot(x0.T, x0) + np.eye(p) * 0.00001), x0.T)
    # bw_beta = np.array([[-0.34144813, -0.41036912, 0.04715828, 0.23441434],
    #                    [0.33240948, -0.81040147, 0.4601485, -0.47714486]])
    sm_y, bw_beta, _ = mvcm(coord_mat, y0)
    res_y0 = y0 * 0
    b_0 = np.zeros(shape=(p, n_v, m))
    for j in range(m):
        b_0[:, :, j] = np.dot(c_mat, sm_y[:, :, j])
        res_y0[:, :, j] = y0[:, :, j] - np.dot(x0, b_0[:, :, j])
    print("""\n Run smoothing on individual functions\n """)
    sm_eta, _, res_eta = mvcm(coord_mat, res_y0, bw_beta, 1)
    sigma_eta = np.zeros(shape=(n_v, m, m))
    omega_eps = np.zeros(shape=(n_v, m, m))
    for l in range(n_v):
        sigma_eta[l, :, :] = np.dot(np.squeeze(sm_eta[:, l, :]).T, np.squeeze(sm_eta[:, l, :])) / (n-sum(z))
        omega_eps[l, :, :] = np.dot(np.squeeze(res_eta[:, l, :]).T, np.squeeze(res_eta[:, l, :])) / (n-sum(z))

    """++++++++++++++++++++++++++++++++++++"""
    print("""\n Save results\n """)
    beta_file_name = output_dir + phase + "/beta.mat"
    savemat(beta_file_name, dict([('beta', b_0)]))
    sigma_file_name = output_dir + phase + "/sigma.mat"
    savemat(sigma_file_name, dict([('sigma_eta', sigma_eta), ('omega_eps', omega_eps)]))
