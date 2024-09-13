from coulomb_stress.coulomb_stress_dynamic import *

if __name__ == "__main__":
    field_points_ = np.load(
        "/home/zjc/seis_data/Turkey1/rup_model_xyy/sub_faults_plane_exp5.npy"
    )
    field_fms_ = np.load(
        "/home/zjc/seis_data/Turkey1/rup_model_xyy/sub_fms_plane_exp5.npy"
    )
    for i in range(6, 7):
        sub_faults_i = np.load(
            "/home/zjc/seis_data/Turkey1/rup_model_xyy/sub_faults_plane_exp%d.npy" % i
        )
        field_points_ = np.concatenate([field_points_, sub_faults_i], axis=0)
        sub_fms_i = np.load(
            "/home/zjc/seis_data/Turkey1/rup_model_xyy/sub_fms_plane_exp%d.npy" % i
        )
        field_fms_ = np.concatenate([field_fms_, sub_fms_i], axis=0)

    path_green_ = "/e/qb"
    path_output_ = "/home/zjc/seis_data/Turkey1/qssp_cfs/cfs_S12_S67_d10/"
    points_geo_ = create_points(dist_range=[0, 320], delta_dist=10)
    prepare_multi_points(
        path_output=path_output_,
        processes_num=8,
        path_faults_source="/home/zjc/seis_data/Turkey1/rup_model_xyy/",
        path_green=path_green_,
        source_inds=[0, 1],
        field_points=field_points_,
        field_fms=field_fms_,
        points_green_geo=points_geo_,
        event_dep_list=[2 * h + 1 for h in range(15)],
        receiver_dep_list=[2 * h + 1 for h in range(15)],
        srate_stf=2,
        srate_cfs=1,
        N_T=128,
        time_reduction=10,
        mu=0.4,
        mu_pore=0.6,
        B_pore=0.75,
        interp=False,
    )
