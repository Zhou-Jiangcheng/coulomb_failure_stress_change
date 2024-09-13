import os
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "Arial",
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def find_pkl(lat, lon, dep, pkl_list):
    for i in range(len(pkl_list)):
        fname = pkl_list[i]
        _, _, lat_i, lon_i, dep_i = fname[:-4].split("_")
        if lat_i == "%f" % lat and lon_i == "%f" % lon and dep_i == "%f" % dep:
            return pkl_list[i]
    else:
        raise ValueError("no matched pkl file!")


def read_stress_results(path_output, path_obs_faults, inds_obs):
    pkl_list = glob.glob(os.path.join(path_output, "*.pkl"))
    pkl_list = [os.path.basename(pkl_file) for pkl_file in pkl_list]
    pkl_list = sorted(pkl_list)[:-1]
    for ind in inds_obs:
        print(ind)
        cfs_plane = []
        normal_stress_plane = []
        shear_stress_plane = []
        sub_faults_obs = np.load(
            os.path.join(path_obs_faults, "sub_faults_plane_exp%d.npy" % ind)
        )
        for i in range(len(sub_faults_obs)):
            lat, lon, dep = list(sub_faults_obs[i])
            # print(i, lat, lon, dep)
            fname = find_pkl(lat, lon, dep, pkl_list)
            with open(os.path.join(path_output, fname), "rb") as fr:
                (
                    stress_enz,
                    sigma_vector,
                    sigma,
                    tau,
                    mean_stress,
                    coulomb_stress,
                    coulomb_stress_pore,
                ) = pickle.load(fr)
            cfs_plane.append(coulomb_stress)
            normal_stress_plane.append(sigma)
            shear_stress_plane.append(tau)
        cfs_plane = np.array(cfs_plane)
        normal_stress_plane = np.array(normal_stress_plane)
        shear_stress_plane = np.array(shear_stress_plane)
        np.save(str(os.path.join(path_output, "cfs_plane_exp%d.npy" % ind)), cfs_plane)
        np.save(str(os.path.join(path_output,
                "normal_stress_plane_exp%d.npy" % ind)), normal_stress_plane)
        np.save(str(os.path.join(path_output,
                "shear_stress_plane_exp%d.npy" % ind)), shear_stress_plane)


if __name__ == "__main__":
    path_output_ = "/home/zjc/seis_data/Turkey1/qssp_cfs/cfs_S12_S5_d10/"
    inds_obs_ = [4]

    # path_output_ = "/home/zjc/seis_data/Turkey1/qssp_cfs/cfs_S12_S678_d10_h/"
    # inds_obs_=[5, 6, 7]
    #
    # path_output_ = "/home/zjc/seis_data/Turkey1/qssp_cfs/cfs_S127_S68_d10_h/"
    # inds_obs_ = [5, 7]

    # path_output_ = "/home/zjc/seis_data/Turkey1/qssp_cfs/cfs_S1278_S9_d10_h/"
    # inds_obs_ = [8]

    path_obs_faults_ = "/home/zjc/seis_data/Turkey1/rup_model_xyy/"
    read_stress_results(
        path_output=path_output_, path_obs_faults=path_obs_faults_, inds_obs=inds_obs_
    )
