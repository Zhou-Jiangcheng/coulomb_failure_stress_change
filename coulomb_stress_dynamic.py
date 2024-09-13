import os
import pickle

import numpy as np
from mpi4py import MPI

from pygrnwang.read_qssp import read_stress_tensor
from pygrnwang.utils import group
from focal_mechanism import plane2nd, plane2mt
from signal_process import resample
from coulomb_stress_static import (
    cal_coulomb_stress,
    cal_coulomb_stress_poroelasticity,
)

d2m = 111194.92664455874


def find_nearest_dep(dep, dep_list):
    return dep_list[np.argmin(np.abs(dep - dep_list))]


def cal_stress_vector_ned(stress_enz, n):
    # 重构3x3应力张量
    sigma11, sigma12, sigma13, sigma22, sigma23, sigma33 = stress_enz.T
    stress_tensor_ned = np.array(
        [
            [sigma22, sigma12, -sigma23],
            [sigma12, sigma11, -sigma13],
            [-sigma23, -sigma13, sigma33],
        ]
    ).T  # Shape will be (n, 3, 3)

    # Perform the dot product of each tensor with the vector n
    sigma_vector = np.einsum("ijk,k->ij", stress_tensor_ned, n.flatten())
    # sigmas = np.dot(sigma_vectors, np.array([n]).T)
    # taus = np.dot(sigma_vectors, np.array([d]).T)
    return sigma_vector


def cal_coulomb_stress_single_point(
        path_green,
        path_faults,
        source_inds,
        field_point,
        points_green_geo,
        event_dep_list,
        receiver_dep_list,
        srate_stf,
        srate_cfs,
        N_T,
        time_reduction,
        n_obs,
        d_obs,
        mu=0.4,
        mu_pore=0.6,
        B_pore=0.75,
        interp=False,
):
    receiver_depth = find_nearest_dep(field_point[2], receiver_dep_list)

    stress_enz = np.zeros((N_T, 6))

    for ind in source_inds:
        # print(ind)

        sub_faults_source = np.load(
            os.path.join(path_faults, "sub_faults_plane%d.npy" % ind)
        )
        sub_fms = np.load(os.path.join(
            path_faults, "sub_fms_plane%d.npy" % ind))
        sub_stfs = np.load(os.path.join(
            path_faults, "sub_stfs_plane%d.npy" % ind))
        sub_m0s = np.load(os.path.join(
            path_faults, "sub_m0s_plane%d.npy" % ind))
        # sub_slips = np.load(os.path.join(path_faults, "sub_slips_plane%d.npy" % ind))

        for i in range(sub_faults_source.shape[0]):
            event_depth = find_nearest_dep(
                sub_faults_source[i][2], event_dep_list)
            stress_enz_1source = read_stress_tensor(
                path_green=path_green,
                event_depth=event_depth,
                receiver_depth=receiver_depth,
                points_green_geo=points_green_geo,
                source=sub_faults_source[i],
                station=field_point,
                mt=plane2mt(1, *sub_fms[i]),
                interp=interp,
            )  # enz

            for i_enz in range(6):
                sub_stf = resample(
                    sub_stfs[i],
                    srate_old=srate_stf,
                    srate_new=srate_cfs,
                    zero_phase=True,
                )
                sub_stf = sub_stf / (np.sum(sub_stf) / srate_cfs) * sub_m0s[i]
                sigma_temp = stress_enz_1source[
                    round(time_reduction * srate_cfs):, i_enz]
                sigma_temp = sigma_temp - sigma_temp[0]

                # sigma_temp[0] = 0
                #
                # len(sub_stf) must larger than time_reduction !!!
                #
                stress_enz_1source[:, i_enz] = (
                    np.convolve(sub_stf, sigma_temp)[:N_T] / srate_cfs
                )
                point_sta = np.array(
                    field_point[:2]) - np.array(sub_faults_source[i][:2])
                dist = np.sqrt(point_sta[0] ** 2 +
                               point_sta[1] ** 2) * d2m / 1e3
                t_cut_slowness = dist * 0.4  # max slowness
                ind_const = round((t_cut_slowness + 1) *
                                  srate_cfs + len(sub_stf))
                if ind_const < N_T:
                    stress_enz_1source[ind_const:,
                                       i_enz] = stress_enz_1source[ind_const, i_enz]
            stress_enz = stress_enz + stress_enz_1source

    n = np.array([n_obs.flatten()]).T
    d = np.array([d_obs.flatten()]).T
    sigma_vector = cal_stress_vector_ned(stress_enz, n)  # ned
    sigma = np.dot(sigma_vector, np.array([n]).T).flatten()
    tau = np.dot(sigma_vector, np.array([d]).T).flatten()
    mean_stress = (stress_enz[:, 0] + stress_enz[:, 3] + stress_enz[:, 5]) / 3
    coulomb_stress = cal_coulomb_stress(
        norm_stress_drop=sigma, shear_stress_drop=tau, mu=mu
    )
    coulomb_stress_pore = cal_coulomb_stress_poroelasticity(
        norm_stress_drop=sigma,
        shear_stress_drop=tau,
        mean_stress_drop=mean_stress,
        mu=mu_pore,
        B=B_pore,
    )

    return (
        stress_enz,
        sigma_vector,
        sigma,
        tau,
        mean_stress,
        coulomb_stress,
        coulomb_stress_pore,
    )


def prepare_multi_points(
        path_output,
        processes_num,
        path_green,
        path_faults_source,
        source_inds,
        field_points,
        field_fms,
        points_green_geo,
        event_dep_list,
        receiver_dep_list,
        srate_stf,
        srate_cfs,
        N_T,
        time_reduction,
        mu=0.4,
        mu_pore=0.6,
        B_pore=0.75,
        interp=False,
):
    N_points = len(field_points)
    paras_list = []
    for i in range(N_points):
        n_obs, d_obs = plane2nd(*field_fms[i])
        paras_list.append(
            [
                path_green,
                path_faults_source,
                source_inds,
                field_points[i],
                points_green_geo,
                event_dep_list,
                receiver_dep_list,
                srate_stf,
                srate_cfs,
                N_T,
                time_reduction,
                n_obs,
                d_obs,
                mu,
                mu_pore,
                B_pore,
                interp,
            ]
        )
    group_list = group(paras_list, processes_num)
    with open(os.path.join(path_output, "group_list.pkl"), "wb") as fw:
        pickle.dump(group_list, fw)


def cal_coulomb_stress_multi_points_mpi(
        path_output,
):
    with open(os.path.join(path_output, "group_list.pkl"), "rb") as fr:
        group_list = pickle.load(fr)
    N_all = 0
    for ind_group in range(len(group_list)):
        N_all = N_all + len(group_list[ind_group])
    for ind_group in range(len(group_list)):
        comm = MPI.COMM_WORLD
        processes_num = comm.Get_size()
        rank = comm.Get_rank()
        if processes_num < len(group_list[0]):
            raise ValueError(
                "processes_num is %d, item num in group is %d. \n"
                "Pleasse check the process num!" % (
                    processes_num, len(group_list[0]))
            )
        print("ind_group:%d rank:%d" % (ind_group, rank))
        if ind_group * len(group_list[0]) + rank < N_all:
            paras = group_list[ind_group][rank]
            cfs_data = cal_coulomb_stress_single_point(*paras)
            with open(
                    os.path.join(
                        path_output,
                        "%d_%d_%f_%f_%f.pkl"
                        % (ind_group, rank, paras[3][0], paras[3][1], paras[3][2]),
                    ),
                    "wb",
            ) as fw:
                pickle.dump(cfs_data, fw)


if __name__ == "__main__":
    pass
