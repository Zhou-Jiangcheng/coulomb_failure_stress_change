import os
import shutil
import subprocess
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import pandas as pd

from pygrnwang.geo import convert_sub_faults_geo2ned
from pygrnwang.focal_mechanism import plane2nd
from pygrnwang.others import cal_max_dist_from_2d_points

d2m = 111194.92664455874


def convert_earth_model_nd2edgrn_inp(path_nd):
    with open(path_nd, "r") as fr:
        lines = fr.readlines()
    lines_new = []
    for i in range(len(lines)):
        temp = lines[i].split()
        if len(temp) > 1:
            for j in range(4):
                temp[j] = "%.1f" % (float(temp[j]) * 1000)
            lines_new.append(temp[:4])
    for i in range(len(lines_new)):
        # print(lines_new[i])
        lines_new[i] = "  ".join([str(i + 1)] + lines_new[i]) + "\n"
    return lines_new


def cal_source_inp(
    path_faults_sources,
    source_plane_inds,
    source_ref,
    sub_len,
    path_output,
    time_cut=None,
):
    """

    :param path_faults_sources:
    :param source_plane_inds:
    :param source_ref: [deg,deg,m]
    :param sub_len: m
    :param path_output:
    :param time_cut: [time, srate] [s,Hz]
    :return:
    """
    slips_list = []
    refs_list = []
    fms_list = []
    num_sources_list = []
    # for i in range(num_planes):
    for i in source_plane_inds:
        if time_cut is None:
            sub_slips = np.load(
                os.path.join(path_faults_sources,
                             "sub_slips_plane_exp%d.npy" % i)
            )
        else:
            sub_slips = np.load(
                os.path.join(path_faults_sources,
                             "sub_slips_plane_exp%d.npy" % i)
            )
            sub_m0s = np.load(
                os.path.join(path_faults_sources,
                             "sub_m0s_plane_exp%d.npy" % i)
            )
            mu_A = sub_m0s / sub_slips
            sub_stfs = np.load(
                os.path.join(path_faults_sources,
                             "sub_stfs_plane_exp%d.npy" % i)
            )
            sub_slips = (
                np.sum(
                    sub_stfs[:, : round(time_cut[0] * time_cut[1])],
                    axis=1,
                )
                / mu_A
                / time_cut[1]
            )
        slips_list.append(sub_slips)

        sub_faults = np.load(
            os.path.join(path_faults_sources, "sub_faults_plane_exp%d.npy" % i)
        )
        sub_faults[:, 2] = sub_faults[:, 2] * 1e3
        sub_faults = convert_sub_faults_geo2ned(
            sub_faults=sub_faults, source_point=source_ref
        )
        sub_faults[:, 2] = sub_faults[:, 2] + source_ref[2]
        sub_faults[sub_faults[:, 2] <= 0, 2] = 0

        refs_list.append(sub_faults)

        sub_fms = np.load(
            os.path.join(path_faults_sources, "sub_fms_plane_exp%d.npy" % i)
        )
        fms_list.append(sub_fms)

        num_sources_list.append(len(sub_slips))

    num_sources = sum(num_sources_list)
    sources = np.zeros((num_sources, 10))
    for i in range(len(source_plane_inds)):
        ind_start = sum(num_sources_list[:i])
        ind_end = sum(num_sources_list[: i + 1])
        sources[ind_start:ind_end, 1] = slips_list[i].flatten()
        sources[ind_start:ind_end, 2:5] = refs_list[i]
        sources[ind_start:ind_end, 5] = np.ones(ind_end - ind_start) * sub_len
        sources[ind_start:ind_end, 6] = np.ones(ind_end - ind_start) * sub_len
        sources[ind_start:ind_end, 7:] = fms_list[i]
    sources[:, 0] = np.arange(1, num_sources + 1)
    # print(sources)
    np.save(os.path.join(path_output, "sources.npy"), sources)
    return sources


def cal_obs_inp(
    path_faults_obs,
    obs_ref,
    sub_len,
    path_output,
    lat_range=None,
    lon_range=None,
):
    if lat_range is None:
        sub_faults = np.load(os.path.join(path_faults_obs, "sub_faults.npy"))
        sub_faults[:, 2] = sub_faults[:, 2] * 1e3
        sub_faults = convert_sub_faults_geo2ned(
            sub_faults=sub_faults, source_point=obs_ref
        )
        x_start = np.min(np.floor(sub_faults[:, 0]))
        x_end = np.max(np.ceil(sub_faults[:, 0]))
        y_start = np.min(np.floor(sub_faults[:, 1]))
        y_end = np.max(np.ceil(sub_faults[:, 1]))
    else:
        x_start = (lat_range[0] - obs_ref[0]) * d2m
        x_end = (lat_range[1] - obs_ref[0]) * d2m
        y_start = (lon_range[0] - obs_ref[1]) * d2m
        y_end = (lon_range[1] - obs_ref[1]) * d2m

    nx = round(np.ceil((x_end - x_start) / sub_len) + 1)
    ny = round(np.ceil((y_end - y_start) / sub_len) + 1)
    obs = np.array(
        [
            [nx, x_start, x_end],
            [ny, y_start, y_end],
        ]
    )
    np.save(os.path.join(path_output, "obs.npy"), obs)
    return obs


def cal_grn_inp(path_output, sub_len, rmax=None):
    if rmax is None:
        sources = np.load(os.path.join(path_output, "sources.npy"))
        obs = np.load(os.path.join(path_output, "obs.npy"))
        x_obs = np.linspace(obs[0, 1], obs[0, 2], round(obs[0, 0]))
        y_obs = np.linspace(obs[1, 1], obs[1, 2], round(obs[1, 0]))
        X_obs, Y_obs = np.meshgrid(x_obs, y_obs)
        points_source = sources[:, 2:4]
        points_obs = np.array([X_obs.flatten(), Y_obs.flatten()]).T
        rmax = cal_max_dist_from_2d_points(points_source, points_obs)

    rmax = round(np.ceil(rmax / 1000)) * 1000
    nr = round(np.ceil(rmax / sub_len)) + 1
    rmin = 0

    sources_grn = np.array([nr, rmin, rmax])
    np.save(os.path.join(path_output, "sources_grn.npy"), sources_grn)
    return sources_grn


def create_edgrn_inp(
    obs_dep,
    path_output,
    source_dep_list,
    path_nd=None,
    earth_model_layer_num=None,
):
    sources_grn = np.load(os.path.join(path_output, "sources_grn.npy"))
    path_edgrn_inp_output = os.path.join(path_output, "edgrn", "edgrn.inp")
    if os.path.exists(path_edgrn_inp_output):
        with open(path_edgrn_inp_output, "r") as fr:
            lines = fr.readlines()
    else:
        from coulomb_failure_stress_change.edgrn_inp import s

        lines = s.split("\n")
        lines = [line + "\n" for line in lines]
    lines_new = lines.copy()
    lines_new[39] = "%.1f\n" % obs_dep
    lines_new[40] = "%d %f %f\n" % (
        sources_grn[0], sources_grn[1], sources_grn[2])
    lines_new[41] = "%d %f %f\n" % (
        len(source_dep_list),
        source_dep_list[0],
        source_dep_list[-1],
    )
    lines_new[61] = " './' '%.1f.ss' '%.1f.ds' '%.1f.cl'\n" % (
        obs_dep,
        obs_dep,
        obs_dep,
    )
    if path_nd is not None:
        lines_earth = convert_earth_model_nd2edgrn_inp(path_nd=path_nd)
    if earth_model_layer_num is not None:
        lines_earth = lines_earth[:earth_model_layer_num]
    else:
        earth_model_layer_num = len(lines_earth)
    lines_new[-(earth_model_layer_num + 1): -1] = lines_earth

    with open(
        os.path.join(path_output, "edgrn", "edgrn_%.1f.inp" % obs_dep), "w"
    ) as fw:
        fw.writelines(lines_new)


def create_edcmp_inp(
    obs_dep,
    hs_flag,
    path_output,
    lam=3e10,
    mu=3e10,
):
    """

    :param obs_dep: m
    :param hs_flag: 0/1
    :param path_output
    :param lam: N/M^2
    :param mu: N/M^2
    :return:
    """
    sources = np.load(os.path.join(path_output, "sources.npy"))
    obs = np.load(os.path.join(path_output, "obs.npy"))
    sources_str = ""
    for i in range(len(sources)):
        sources_str += "%d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % (
            int(sources[i, 0]),
            sources[i, 1],
            sources[i, 2],
            sources[i, 3],
            sources[i, 4],
            sources[i, 5],
            sources[i, 6],
            sources[i, 7],
            sources[i, 8],
            sources[i, 9],
        )
    path_output_inp_output = os.path.join(path_output, "edcmp", "edcmp.inp")
    if os.path.exists(path_output_inp_output):
        with open(path_output_inp_output, "r") as fr:
            lines = fr.readlines()
    else:
        from coulomb_failure_stress_change.edcmp_inp import s

        lines = s.split("\n")
        lines = [line + "\n" for line in lines]
    lines_new = lines.copy()
    lines_new[54] = "%d %f %f\n" % (obs[0, 0], obs[0, 1], obs[0, 2])
    lines_new[55] = "%d %f %f\n" % (obs[1, 0], obs[1, 1], obs[1, 2])
    lines_new[70] = "'%.1f.disp' '%.1f.strn' '%.1f.strss' '%.1f.tilt'\n" % (
        obs_dep,
        obs_dep,
        obs_dep,
        obs_dep,
    )
    lines_new[100] = "%d\n" % len(sources)
    lines_new[105] = sources_str
    lines_new[138] = "%d\n" % hs_flag
    if hs_flag == 0:
        lines_new[139] = "%.1f %f %f\n" % (obs_dep, lam, mu)
    elif hs_flag == 1:
        lines_new[139] = "'../edgrn/'  '%.1f.ss'  '%.1f.ds'  '%.1f.cl'\n" % (
            obs_dep,
            obs_dep,
            obs_dep,
        )
    else:
        raise ValueError("hs_flag wrong")
    with open(
        os.path.join(path_output, "edcmp", "edcmp_%.1f.inp" % obs_dep), "w"
    ) as fw:
        fw.writelines(lines_new)


def prepare_cfs_static(
    path_output,
    path_bin_edgrn,
    path_bin_edcmp,
    path_faults_sources,
    source_plane_inds,
    sub_length_source,
    source_dep_list,
    path_faults_obs,
    sub_length_obs,
    obs_dep_list,
    lat_range,
    lon_range,
    rmax_grn,
    ref_point,
    hs_flag,
    path_nd=None,
    earth_model_layer_num=None,
    lam=None,
    mu=None,
):
    os.makedirs(path_output, exist_ok=True)
    os.makedirs(os.path.join(path_output, "edgrn"), exist_ok=True)
    os.makedirs(os.path.join(path_output, "edcmp"), exist_ok=True)
    shutil.copy(path_bin_edgrn, path_output)
    shutil.copy(path_bin_edcmp, path_output)
    cal_source_inp(
        path_faults_sources=path_faults_sources,
        source_plane_inds=source_plane_inds,
        source_ref=ref_point,
        sub_len=sub_length_source,
        path_output=path_output,
    )
    cal_obs_inp(
        path_faults_obs=path_faults_obs,
        obs_ref=ref_point,
        sub_len=sub_length_obs,
        path_output=path_output,
        lat_range=lat_range,
        lon_range=lon_range,
    )
    cal_grn_inp(
        path_output=path_output,
        sub_len=sub_length_source,
        rmax=rmax_grn,
    )

    for dep in obs_dep_list:
        create_edgrn_inp(
            obs_dep=dep,
            path_output=path_output,
            source_dep_list=source_dep_list,
            path_nd=path_nd,
            earth_model_layer_num=earth_model_layer_num,
        )

    for dep in obs_dep_list:
        create_edcmp_inp(
            obs_dep=dep,
            hs_flag=hs_flag,
            path_output=path_output,
            lam=lam,
            mu=mu,
        )


def create_static_cfs_lib(
    processes_num,
    path_output,
    path_faults_obs,
    obs_plane_inds,
    sub_length_obs,
    obs_dep_list,
    ref_point,
    hs_flag,
):
    if hs_flag == 1:
        call_edgrn_parallel_single_node(
            obs_dep_list=obs_dep_list,
            path_output=path_output,
            processes_num=processes_num,
        )

    call_edcmp_parallel_single_node(
        obs_dep_list=obs_dep_list,
        path_output=path_output,
        processes_num=processes_num,
    )
    read_stress(path_output, obs_dep_list)
    project_to_obs_faults(
        path_output=path_output,
        path_faults_obs=path_faults_obs,
        obs_plane_inds=obs_plane_inds,
        obs_ref=ref_point,
        sub_length=sub_length_obs,
        dep_list=obs_dep_list,
    )


def cal_static_coulomb_stress(path_output, obs_plane_inds, mu_f=0.4, B=0.75):
    cal_all_coulomb_stress(
        path_output=path_output,
        obs_plane_inds=obs_plane_inds,
        mu_f=mu_f,
    )
    cal_all_coulomb_stress_poroelasticity(
        path_output=path_output,
        obs_plane_inds=obs_plane_inds,
        mu_f=mu_f,
        B=B,
    )


def call_edgrn(obs_dep, path_output):
    # print(obs_dep)
    os.chdir(os.path.join(path_output, "edgrn"))
    path_inp = str(os.path.join(
        path_output, "edgrn", "edgrn_%.1f.inp" % obs_dep))
    edcmp_process = subprocess.Popen(
        [os.path.join(path_output, "edgrn2.0")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    edcmp_process.communicate(str.encode(path_inp))


def call_edcmp(obs_dep, path_output):
    # print(obs_dep)
    os.chdir(os.path.join(path_output, "edcmp"))
    path_inp = str(os.path.join(
        path_output, "edcmp", "edcmp_%.1f.inp" % obs_dep))
    edcmp_process = subprocess.Popen(
        [os.path.join(path_output, "edcmp2.0")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    edcmp_process.communicate(str.encode(path_inp))


def _call_edgrn(args):
    call_edgrn(*args)


def call_edgrn_parallel_single_node(obs_dep_list, path_output, processes_num):
    print("call edgrn")
    input_list = []
    for i in range(len(obs_dep_list)):
        input_list.append([obs_dep_list[i], path_output])
    with mp.Pool(processes=processes_num) as pool:
        list(tqdm(pool.imap(_call_edgrn, input_list), total=len(input_list)))


def _call_edcmp(args):
    call_edcmp(*args)


def call_edcmp_parallel_single_node(obs_dep_list, path_output, processes_num):
    print("call edcmp")
    input_list = []
    for i in range(len(obs_dep_list)):
        input_list.append([obs_dep_list[i], path_output])
    with mp.Pool(processes=processes_num) as pool:
        list(tqdm(pool.imap(_call_edcmp, input_list), total=len(input_list)))


def read_stress(path_output, dep_list):
    obs = np.load(os.path.join(path_output, "obs.npy"))
    print(obs)
    stress_all = np.zeros((round(obs[0, 0] * obs[1, 0]), 8, len(dep_list)))
    for i in range(len(dep_list)):
        # X_m Y_m Sxx_Pa Syy_Pa Szz_Pa Sxy_Pa Syz_Pa Szx_Pa
        df = pd.read_csv(
            str(os.path.join(path_output, "edcmp",
                "%.1f.strss" % dep_list[i])),
            skiprows=3,
            sep="\\s+",
            header=None,
        )
        stress_all[:, :, i] = df.to_numpy()
    np.save(os.path.join(path_output, "stress_all.npy"), stress_all)
    return stress_all


def project_to_obs_faults(
    path_output,
    path_faults_obs,
    obs_plane_inds,
    obs_ref,
    sub_length,
    dep_list,
):
    dep_list = np.array(dep_list)

    def _find_xy(x_in, y_in, x_start, y_start, nx):
        num_x = np.floor((x_in - x_start) / sub_length)
        num_y = np.floor((y_in - y_start) / sub_length)
        ind = round(num_y * nx + num_x)
        return ind

    def _find_z(z_in):
        ind = round(np.argmin(np.abs(z_in - dep_list)))
        return ind

    obs = np.load(os.path.join(path_output, "obs.npy"))
    stress_all = np.load(os.path.join(path_output, "stress_all.npy"))

    for i in obs_plane_inds:
        sub_faults = np.load(
            os.path.join(path_faults_obs, "sub_faults_plane_exp%d.npy" % i)
        )
        sub_fms = np.load(os.path.join(
            path_faults_obs, "sub_fms_plane_exp%d.npy" % i))
        print(sub_faults.shape)
        sub_faults[:, 2] = sub_faults[:, 2] * 1e3

        sub_faults = convert_sub_faults_geo2ned(
            sub_faults=sub_faults, source_point=obs_ref
        )
        sub_faults[:, 2] = sub_faults[:, 2] + obs_ref[2]

        sub_norm_stress = np.zeros(len(sub_faults))
        sub_shear_stress = np.zeros(len(sub_faults))
        sub_mean_stress = np.zeros(len(sub_faults))
        for j in range(len(sub_faults)):
            x, y, z = sub_faults[j, :].tolist()
            n, d = plane2nd(*sub_fms[j, :])
            n = np.array([n.flatten()]).T
            d = np.array([d.flatten()]).T
            ind_xy = _find_xy(
                x_in=x, y_in=y, x_start=obs[0,
                                            1], y_start=obs[1, 1], nx=obs[0, 0]
            )
            ind_z = _find_z(z_in=z)
            stress = stress_all[ind_xy, 2:, ind_z]
            # Sxx_Pa Syy_Pa Szz_Pa Sxy_Pa Syz_Pa Szx_Pa

            # nn ee dd ne ed nd
            # ned
            sigma_tensor = np.array(
                [
                    [stress[0], stress[3], stress[5]],
                    [stress[3], stress[1], stress[4]],
                    [stress[5], stress[4], stress[2]],
                ]
            )

            sigma_vector = np.dot(sigma_tensor, n)
            sigma = np.dot(sigma_vector.T, n)[0][0]
            # tau = np.linalg.norm(sigma_vector - sigma * n)
            tau = np.dot(sigma_vector.T, d)[0][0]
            mean_stress = np.sum(stress[:3]) / 3

            sub_norm_stress[j] = sigma
            sub_shear_stress[j] = tau
            sub_mean_stress[j] = mean_stress
        np.save(
            str(os.path.join(path_output, "norm_stress_%d.npy" % i)), sub_norm_stress
        )
        np.save(
            str(os.path.join(path_output, "shear_stress_%d.npy" % i)), sub_shear_stress
        )
        np.save(
            str(os.path.join(path_output, "mean_stress_%d.npy" % i)), sub_mean_stress
        )
        np.save(
            str(os.path.join(path_output, "sub_faults_plane%d.npy" % i)), sub_faults
        )


def cal_coulomb_stress_one_dep(
    path_output,
    strike,
    dip,
    rake,
    obs_dep,
    obs_dep_list,
    mu=0.4,
):
    stress_all = np.load(os.path.join(path_output, "stress_all.npy"))
    ind = round(np.argmin(np.abs(obs_dep - np.array(obs_dep_list))))
    n, d = plane2nd(strike=strike, dip=dip, rake=rake)
    n = n.flatten()
    d = d.flatten()
    # print(n, d)
    stress_components = stress_all[:, -6:, ind]
    # Reshape and reorganize the stress tensor components
    # The new shape will be (n, 3, 3) corresponding to n 3x3 tensors
    stress_tensors = np.zeros((stress_components.shape[0], 3, 3))
    stress_tensors[:, 0, 0] = stress_components[:, 0]  # sxx
    stress_tensors[:, 1, 1] = stress_components[:, 1]  # syy
    stress_tensors[:, 2, 2] = stress_components[:, 2]  # szz
    stress_tensors[:, 0, 1] = stress_tensors[:, 1, 0] = stress_components[
        :, 3
    ]  # sxy = syx
    stress_tensors[:, 1, 2] = stress_tensors[:, 2, 1] = stress_components[
        :, 4
    ]  # syz = szy
    stress_tensors[:, 0, 2] = stress_tensors[:, 2, 0] = stress_components[
        :, 5
    ]  # sxz = szx

    # Perform the dot product of each tensor with the vector n
    sigma_vectors = np.einsum("ijk,k->ij", stress_tensors, n)

    sigmas = np.dot(sigma_vectors, np.array([n]).T)
    taus = np.dot(sigma_vectors, np.array([d]).T)

    coulomb_stress = cal_coulomb_stress(
        norm_stress_drop=sigmas, shear_stress_drop=taus, mu=mu
    )
    coulomb_stress = np.concatenate(
        [
            np.array([strike]),
            np.array([dip]),
            np.array([rake]),
            coulomb_stress.flatten(),
        ]
    )

    np.save(
        str(os.path.join(path_output, "sigma_vectors_dep%.1f" %
            obs_dep_list[ind])),
        sigma_vectors,
    )
    np.save(
        str(os.path.join(path_output, "sigmas_dep%.1f" %
            obs_dep_list[ind])), sigmas
    )
    np.save(str(os.path.join(path_output, "taus_dep%.1f" %
            obs_dep_list[ind])), taus)
    np.save(
        str(os.path.join(path_output, "coulomb_stress_dep%.1f" %
            obs_dep_list[ind])),
        coulomb_stress,
    )
    return coulomb_stress


def cal_coulomb_stress(
    norm_stress_drop,
    shear_stress_drop,
    mu_f=0.4,
):
    """
    :param norm_stress_drop:
    :param shear_stress_drop:
    :param mu_f: effective coefficient of friction
    :return:
    """
    coulomb_stress = shear_stress_drop + mu_f * norm_stress_drop
    return coulomb_stress


def cal_coulomb_stress_poroelasticity(
    norm_stress_drop,
    shear_stress_drop,
    mean_stress_drop,
    mu_f=0.6,
    B=0.75,
):
    """

    :param norm_stress_drop:
    :param shear_stress_drop:
    :param mean_stress_drop:
    :param mu_f: coefficient of friction
    :param B: Skempton's coefficient
    :return:
    """
    coulomb_stress = shear_stress_drop + mu_f * (
        norm_stress_drop + B * mean_stress_drop
    )
    return coulomb_stress


def cal_all_coulomb_stress(path_output, obs_plane_inds, mu_f=0.4):
    coulomb_max = 0
    for i in obs_plane_inds:
        print(i)
        norm_stress = np.load(os.path.join(
            path_output, "norm_stress_%d.npy" % i))
        shear_stress = np.load(os.path.join(
            path_output, "shear_stress_%d.npy" % i))
        coulomb_stress = cal_coulomb_stress(
            norm_stress, shear_stress, mu_f=mu_f)
        np.save(
            str(os.path.join(path_output, "coulomb_stress_%d.npy" % i)), coulomb_stress
        )
        # print(coulomb_stress)
        if coulomb_max < np.max(np.abs(coulomb_stress)):
            coulomb_max = np.max(coulomb_stress)
    print(coulomb_max)


def cal_all_coulomb_stress_poroelasticity(
    path_output, obs_plane_inds, mu_f=0.6, B=0.75
):
    coulomb_max = 0
    for i in obs_plane_inds:
        print(i)
        norm_stress = np.load(os.path.join(
            path_output, "norm_stress_%d.npy" % i))
        shear_stress = np.load(os.path.join(
            path_output, "shear_stress_%d.npy" % i))
        mean_stress = np.load(os.path.join(
            path_output, "mean_stress_%d.npy" % i))
        coulomb_stress = cal_coulomb_stress_poroelasticity(
            norm_stress, shear_stress, mean_stress, mu_f=mu_f, B=B
        )
        np.save(
            str(os.path.join(path_output, "coulomb_stress_poroelasticity_%d.npy" % i)),
            coulomb_stress,
        )
        # print(coulomb_stress)
        if coulomb_max < np.max(np.abs(coulomb_stress)):
            coulomb_max = np.max(np.abs(coulomb_stress))
    print(coulomb_max)


if __name__ == "__main__":
    pass
