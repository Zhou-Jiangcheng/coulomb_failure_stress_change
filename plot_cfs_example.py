import os

import numpy as np
from matplotlib.lines import lineStyles
from scipy.ndimage import zoom
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.colors import from_levels_and_colors
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.patheffects as path_effects

from geo import convert_sub_faults_geo2ned

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "Arial",
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def _reshape_sub_faults(sub_faults, num_dip):
    mu_strike = sub_faults[num_dip] - sub_faults[0]
    mu_dip = sub_faults[1] - sub_faults[0]
    sub_faults = sub_faults - mu_strike / 2 - mu_dip / 2
    X: np.ndarray = sub_faults[:, 0]
    Y: np.ndarray = sub_faults[:, 1]
    Z: np.ndarray = sub_faults[:, 2]

    X = X.reshape(-1, num_dip)
    Y = Y.reshape(-1, num_dip)
    Z = Z.reshape(-1, num_dip)

    X = np.concatenate([X, np.array([X[:, -1] + mu_dip[0]]).T], axis=1)
    Y = np.concatenate([Y, np.array([Y[:, -1] + mu_dip[1]]).T], axis=1)
    Z = np.concatenate([Z, np.array([Z[:, -1] + mu_dip[2]]).T], axis=1)

    X = np.concatenate([X, np.array([X[-1, :] + mu_strike[0]])], axis=0)
    Y = np.concatenate([Y, np.array([Y[-1, :] + mu_strike[1]])], axis=0)
    Z = np.concatenate([Z, np.array([Z[-1, :] + mu_strike[2]])], axis=0)

    zoom_x = 4
    zoom_y = 4
    order = 1
    mode = "constant"
    X = zoom(X, [zoom_x, zoom_y], order=order, mode=mode)
    Y = zoom(Y, [zoom_x, zoom_y], order=order, mode=mode)
    Z = zoom(Z, [zoom_x, zoom_y], order=order, mode=mode)

    return X, Y, Z, zoom_x, zoom_y


def plot_cfs_sub(
    n_t, N_T, path_output, path_obs_fault, ind_obs, num_dip, cmap, norm, ax
):
    stress_list = []

    sub_faults = np.load(path_obs_fault)
    X, Y, Z, zoom_x, zoom_y = _reshape_sub_faults(
        sub_faults=sub_faults, num_dip=num_dip
    )

    sub_stress = np.load(os.path.join(
        path_output, "cfs_plane_exp%d.npy" % ind_obs))
    # normal_stress = np.load(
    #     os.path.join(path_output, "normal_stress_plane_exp%d.npy" % ind_obs)
    # )
    # shear_stress = np.load(
    #     os.path.join(path_output, "shear_stress_plane_exp%d.npy" % ind_obs)
    # )
    # sub_stress = np.abs(shear_stress) + 0.4 * normal_stress

    # print(
    #     np.min(sub_stress[:, :N_T]) / 1e6,
    #     "MPa",
    #     np.max(sub_stress[:, :N_T]) / 1e6,
    #     "MPa",
    # )
    sub_stress = sub_stress[:, n_t].flatten()
    sub_stress: np.ndarray = sub_stress.reshape(-1, num_dip)
    sub_stress = np.concatenate(
        [
            sub_stress,
            np.array([sub_stress[:, -1]]).T,
        ],
        axis=1,
    )
    sub_stress = np.concatenate(
        [
            sub_stress,
            np.array([sub_stress[-1, :]]),
        ],
        axis=0,
    )
    sub_stress = zoom(sub_stress, [zoom_x, zoom_y], order=3)
    stress_list.append([X, Y, Z, sub_stress])

    X_num = stress_list[0][0].shape[0]
    Y_num = stress_list[0][0].shape[1]
    stress_plot = stress_list[0][-1] / 1e6

    X, Y = np.meshgrid(np.arange(X_num), np.arange(Y_num))
    C = stress_plot
    m_plane = ax.pcolormesh(
        X.T,
        Y.T,
        C,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    ax.invert_yaxis()
    ax.set_aspect(1)
    return ax, m_plane, zoom_x, zoom_y


def plot_cfs_all(
    n_t_list,
    nrows,
    ncols,
    srate,
    path_output,
    path_obs_faults,
    obs_plane_inds,
    num_dip,
    sub_length_km,
    file_name,
):
    h_num = 16
    w_num = 0
    w_num_list = []
    for ind_obs in range(len(path_obs_faults)):
        print(ind_obs)
        sub_faults = np.load(os.path.join(path_obs_faults[ind_obs]))
        sub_stress = np.load(
            os.path.join(path_output, "cfs_plane_exp%d.npy" %
                         obs_plane_inds[ind_obs])
        )
        normal_stress = np.load(
            os.path.join(
                path_output, "normal_stress_plane_exp%d.npy" % obs_plane_inds[ind_obs]
            )
        )
        shear_stress = np.load(
            os.path.join(
                path_output, "shear_stress_plane_exp%d.npy" % obs_plane_inds[ind_obs]
            )
        )
        # sub_stress = np.abs(shear_stress) + 0.4 * normal_stress
        print(
            "shear_stress",
            np.min(shear_stress[:, n_t_list[0]: n_t_list[-1]]) / 1e6,
            "MPa",
            np.max(shear_stress[:, n_t_list[0]: n_t_list[-1]]) / 1e6,
            "MPa",
        )
        print(
            "normal_stress",
            np.min(normal_stress[:, n_t_list[0]: n_t_list[-1]]) / 1e6,
            "MPa",
            np.max(normal_stress[:, n_t_list[0]: n_t_list[-1]]) / 1e6,
            "MPa",
        )
        print(
            "coulomb_stress",
            np.min(sub_stress[:, n_t_list[0]: n_t_list[-1]]) / 1e6,
            "MPa",
            np.max(sub_stress[:, n_t_list[0]: n_t_list[-1]]) / 1e6,
            "MPa",
        )
        w_num_list.append(sub_faults.shape[0] / h_num)
        w_num = w_num + sub_faults.shape[0] / h_num
    w = 22 / 2.54
    h = 22 * (nrows * h_num) / (ncols * w_num) / 2.54

    tick_range_bf10 = [-2, 2]
    tick_range_af10 = [-2, 2]

    fig = plt.figure(figsize=(w / 0.9, h / 0.8))
    for i in range(nrows):
        for j in range(ncols):
            ind = j + i * ncols
            if n_t_list[ind] * srate <= 10:
                cmap = matplotlib.colormaps["seismic"]
                norm = Normalize(
                    vmin=tick_range_bf10[0], vmax=tick_range_bf10[1])
            else:
                cmap = matplotlib.colormaps["seismic"]
                norm = Normalize(
                    vmin=tick_range_af10[0], vmax=tick_range_af10[1])
            sub_w_num = 0
            for k in range(len(obs_plane_inds)):
                ax = fig.add_axes(
                    (
                        j / ncols * 0.9 + 1 / ncols * 0.8 * sub_w_num / w_num + 0.05,
                        (1 - (i + 1) / nrows * 0.8 - 0.02),
                        1 / ncols * 0.8 * w_num_list[k] / w_num,
                        1 / nrows * 0.78,
                    )
                )
                ax, m_plane, zoom_x, zoom_y = plot_cfs_sub(
                    n_t_list[ind],
                    n_t_list[-1],
                    path_output,
                    path_obs_faults[k],
                    obs_plane_inds[k],
                    num_dip,
                    cmap,
                    norm,
                    ax,
                )
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                text = ax.text(
                    xlim[1] - 4 * zoom_x,
                    ylim[0] - 2.5 * zoom_y,
                    "S%d" % (obs_plane_inds[k] + 1),
                    ha="left",
                    va="top",
                    weight="bold",
                    color="black",
                )
                text.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=1, foreground="white"),
                        path_effects.Normal(),
                    ]
                )
                if k == 0:
                    text = ax.text(
                        xlim[0] + 1 * zoom_x,
                        ylim[1] + 1 * zoom_y,
                        "%d s" % (n_t_list[ind] * srate),
                        ha="left",
                        va="top",
                        weight="bold",
                        color="black",
                    )
                    text.set_path_effects(
                        [
                            path_effects.Stroke(
                                linewidth=1, foreground="white"),
                            path_effects.Normal(),
                        ]
                    )
                    if i == nrows - 1:
                        ax.set_xlabel("Along Strike (km)")
                        ax.xaxis.set_label_coords(
                            w_num / w_num_list[0] * 0.5, -0.2)
                    if j == 0:
                        if i % 2 == 1:
                            ax.set_ylabel("Along Dip (km)")
                            ax.yaxis.set_label_coords(-0.4, 1)
                        h_num_zoom = round((h_num + 1) * zoom_y)
                        ticks = [y for y in range(h_num_zoom)]
                        tick_labels = [
                            "%d" % round(y * sub_length_km / zoom_y)
                            for y in range(h_num_zoom)
                        ]
                        gap_10_km = round(
                            h_num_zoom * 10 / ((h_num + 1) * sub_length_km)
                        )
                        ax.set_yticks(ticks[::gap_10_km])
                        ax.set_yticklabels(tick_labels[::gap_10_km])
                    else:
                        ax.set_yticks([])
                else:
                    ax.set_yticks([])
                if i == nrows - 1:
                    sub_w_num_zoom = round((w_num_list[k] + 1) * zoom_x)
                    ticks = [x for x in range(sub_w_num_zoom)]
                    tick_labels = [
                        "%d" % round(x * sub_length_km / zoom_x)
                        for x in range(sub_w_num_zoom)
                    ]
                    gap_10_km = round(
                        sub_w_num_zoom * 10 /
                        ((w_num_list[k] + 1) * sub_length_km)
                    )
                    ax.set_xticks(ticks[::gap_10_km])
                    ax.set_xticklabels(tick_labels[::gap_10_km])
                else:
                    ax.set_xticks([])
                sub_w_num = sub_w_num + w_num_list[k]

    cax = fig.add_axes((0.2, 0.06, 0.6, 0.03))
    m = cm.ScalarMappable(cmap=cmap)
    m.set_clim(tick_range_af10[0], tick_range_af10[1])
    cbar = fig.colorbar(m, cax=cax, orientation="horizontal")
    cbar.set_label("Coulomb Failure Stress Change (MPa)")
    cbar.ax.xaxis.set_label_coords(0.5, -1.3)
    cbar.ax.set_xticks(np.linspace(tick_range_af10[0], tick_range_af10[1], 11))
    cbar.ax.xaxis.set_tick_params(pad=3)

    # cbar_ax_top = cbar.ax.twiny()
    # cbar_ax_top.set_xlim(tick_range_bf10[0], tick_range_bf10[1])
    # cbar_ax_top.set_xticks(np.linspace(tick_range_bf10[0], tick_range_bf10[1], 11))
    # cbar_ax_top.xaxis.set_tick_params(pad=1)
    # cbar_ax_top.xaxis.set_ticks_position('top')
    plt.savefig(file_name + ".pdf")
    plt.savefig(file_name + ".svg")
    plt.savefig(file_name + ".png")
    plt.show()


if __name__ == "__main__":
    path_rup_model = "/home/zjc/seis_data/Turkey1/rup_model_xyy"
    obs_plane_inds_ = [5, 6]
    path_obs_faults_ = []
    for i in obs_plane_inds_:
        path_obs_faults_.append(
            os.path.join(path_rup_model, "sub_faults_plane_exp%d.npy" % i)
        )

    path_output_ = "/home/zjc/seis_data/Turkey1/qssp_cfs/cfs_S12_S67_d10/"
    plot_cfs_all(
        n_t_list=[t for t in range(1, 21)],
        nrows=4,
        ncols=5,
        srate=1,
        path_output=path_output_,
        path_obs_faults=path_obs_faults_,
        obs_plane_inds=obs_plane_inds_,
        num_dip=16,
        sub_length_km=2,
        file_name="cfs_s12_s67_d10",
    )
