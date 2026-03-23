import math

import cmapy
import cv2
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

red = torch.tensor([1., 0., 0.])
blue = torch.tensor([0., 0., 1.])
black = torch.tensor([0, 0, 0])
white = torch.tensor([1., 1., 1.])
gray = torch.tensor([128 / 255, 128 / 255, 128 / 255])
yellow = torch.tensor([1., 1., 0])
green = torch.tensor([0., 128. / 255., 0.])


def calc_cmap(labels):
    max_pixel = np.max(labels)
    min_pixel = np.min(labels)
    delta = max_pixel - min_pixel
    cmap = (labels - min_pixel) / (delta + 1e-6) * 255
    cmap = cmap * (-1)
    cmap = cmap + 255
    cmap = cmap.astype(np.uint8)
    return cmap


def write_obj(points, colors, out_filepath):
    N = points.shape[0]
    fout = open(out_filepath, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %f %f %f\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


def read_obj(filepath):
    values = np.loadtxt(filepath, usecols=(1, 2, 3, 4, 5, 6))
    return values[:, :3], values[:, 3:6]


def read_obj_to_pcd(filepath):
    import open3d
    xyz, rgb = read_obj(filepath)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    pcd.colors = open3d.utility.Vector3dVector(rgb)
    return pcd


def vis_multi_points(points, colors=None, labels=None,
                     opacity=1.0, point_size=10.0, title='title',
                     color_map='Paired', save_fig=False, save_name='example',
                     plot_shape=None, **kwargs):
    """Visualize multiple point clouds at once in splitted windows.

    Args:
        points (list): a list of 2D numpy array.
        colors (list, optional): [description]. Defaults to None.

    Example:
        vis_multi_points([points, pts], labels=[self.sub_clouds_points_labels[cloud_ind], labels])
    """
    import pyvista as pv
    import numpy as np
    from pyvista import themes
    from matplotlib import cm

    my_theme = themes.Theme()
    my_theme.color = 'white'
    my_theme.lighting = True
    my_theme.show_edges = True
    my_theme.edge_color = 'white'
    my_theme.background = 'white'
    my_theme.allow_empty_mesh = True
    my_theme.title = title
    pv.set_plot_theme(my_theme)

    n_clouds = len(points)
    if plot_shape is None:
        plot_shape = (1, n_clouds)
    plotter = pv.Plotter(shape=plot_shape, border=False, **kwargs)

    shape_x, shape_y = plot_shape

    if colors is None:
        colors = [None] * n_clouds
    if labels is None:
        labels = [None] * n_clouds

    idx = -1
    for i in range(shape_x):
        for j in range(shape_y):
            idx += 1
            if idx >= n_clouds:
                break
            plotter.subplot(i, j)
            if len(points[idx].shape) == 3: points[idx] = points[idx][0]
            if colors[idx] is not None and len(colors[idx].shape) == 3: colors[idx] = colors[idx][0]
            if colors[idx] is None and labels[idx] is not None:
                color_maps = cm.get_cmap(color_map, labels[idx].max() + 1)
                colors[idx] = color_maps(labels[idx])[:, :3]
                if colors[idx].min() < 0:
                    colors[idx] = np.array(
                        (colors[idx] - colors[idx].min) / (colors[idx].max() - colors[idx].min()) * 255).astype(
                        np.int8)
            plotter.add_points(points[idx], opacity=opacity, point_size=point_size, render_points_as_spheres=True,
                               scalars=colors[idx], rgb=True, style='points')
    # plotter.link_views() # pyvista might have bug for linked_views. Comment this line out if you cannot see the visualzation result.
    if save_fig:
        plotter.show(screenshot=f'{save_name}.png')
        plotter.close()
    else:
        plotter.show()


def vis_labels(p, label, gt=None, **kwargs):
    vis = kwargs.get('vis', True)
    if gt is None:
        cmap = calc_cmap(label.detach().cpu().numpy())
        colors = torch.from_numpy(cv2.applyColorMap(cmap, cmapy.cmap('cool'))).squeeze()
    else:
        colors = torch.from_numpy(gt).squeeze()
    if p.is_cuda:
        colors = colors.cuda()
    if vis:
        vis_multi_points([p.detach().cpu().numpy()],
                         [colors.detach().cpu().numpy()],
                         plot_shape=(1, 1), **kwargs)
    else:
        return colors


def vis_knn(p, p_idx, group_idx, **kwargs):
    """
    Visualize a point cloud with k-nearest neighbors.
        pc is white, center is blue, neighbors is red.
    :param p: point cloud
    :param p_idx: center point index
    :param group_idx: neighbor point index
    :param kwargs:
    :return:
    """
    vis = kwargs.get('vis', True)
    group_idx = group_idx.long()
    colors = torch.zeros_like(p, dtype=torch.float, device=p.device)
    colors[:] = white
    g_idx = group_idx[p_idx]
    colors[g_idx] = red
    colors[p_idx] = blue
    if vis:
        vis_multi_points(
            [p.detach().cpu().numpy()],
            [colors.detach().cpu().numpy()],
            plot_shape=(1, 1), **kwargs
        )
    else:
        return colors


def vis_projects_3d(p, cam_points, cam_idx, hidden=False, **kwargs):
    """
    Visualize 3d projection results.
    """
    vis = kwargs.get('vis', True)
    orbit_cameras = cam_points.cameras
    depths = cam_points.depths
    ps = []
    cs = []
    n = int(math.sqrt(len(cam_idx)))
    for i in range(n):
        for j in range(n):
            c_idx = cam_idx[i * n + j]
            visible = depths[:, 0, c_idx] != 0
            camera = orbit_cameras[c_idx]
            cmap = calc_cmap(depths[:, 0, c_idx] .detach().cpu().numpy())
            color = torch.from_numpy(cv2.applyColorMap(cmap, cv2.COLORMAP_JET)).squeeze()
            if_visible = torch.nonzero(visible)
            if_visible = if_visible.squeeze(-1)

            # pad camera or target as visible
            if if_visible.shape[0] == 0:
                pad_index = camera.cam_index if camera.cam_index >= 0 else camera.target_index
                if_visible = torch.tensor([pad_index], device=p.device)
            if hidden:
                visible_p = torch.index_select(p, 0, if_visible)
                visible_c = torch.index_select(color, 0, if_visible)
            else:
                visible_p = p
                visible_c = color

            ps.append(visible_p.detach().cpu().numpy())
            cs.append(visible_c.detach().cpu().numpy())
    if vis:
        vis_multi_points(ps, cs, plot_shape=(n, n), **kwargs)
    else:
        return cs


def vis_projects_2d(cam_points, cam_idx, **kwargs):
    """
    Visualize 2d projection results.
    """
    vis = kwargs.get('vis', True)
    axis_off = kwargs.get('axis_off', False)
    save = kwargs.get('save', False)
    uv = cam_points.uv
    depths = cam_points.depths
    delta = torch.round((uv % 1) * 1e5) / 1e5
    xy = uv - delta
    colors = depths
    if vis:
        n = int(math.sqrt(len(cam_idx)))
        fig, axes = plt.subplots(n, n, dpi=640)
        axes_list = []
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes_list.append(axes[i, j])
        for i in range(len(axes_list)):
            if save:
                f, a = plt.subplots(1, 1, dpi=640)
            else:
                f = None
                a = axes_list[i]

            c_idx = cam_idx[i]
            x = xy[:, 0, c_idx]
            y = xy[:, 1, c_idx]
            c = colors[:, 0, c_idx]

            a.set_xticks([])
            a.set_yticks([])
            if axis_off:
                a.spines['right'].set_visible(False)
                a.spines['left'].set_visible(False)
                a.spines['bottom'].set_visible(False)
                a.spines['top'].set_visible(False)
            else:
                a.set_title(f'camera-{c_idx}', loc='left', fontsize=12)

            visible = depths[:, 0, c_idx] != 0
            if_visible = torch.nonzero(visible)
            if_visible = if_visible.squeeze(-1)
            x = torch.index_select(x, 0, if_visible)
            y = torch.index_select(y, 0, if_visible)
            c = torch.index_select(c, 0, if_visible)
            sns.scatterplot(ax=a, x=x, y=y, s=24, c=c, cmap='rainbow')

            if save:
                f.savefig(f'./camera-{c_idx}.png')
        if not save:
            fig.set_size_inches(10 * n // 4, 8 * n // 4)
            fig.show()
    else:
        return xy, colors


if __name__ == '__main__':
    vis_root = './test_datas'
    name = 's3dis'
    idx = 'Area_5_lobby_1'

    rgb = f'{vis_root}/rgb-{name}-{idx}.txt'
    gt = f'{vis_root}/gt-{name}-{idx}.txt'
    pred = f'{vis_root}/pred-{name}-{idx}.txt'

    # input_pcd = read_obj_to_pcd(rgb)
    # gt_pcd = read_obj_to_pcd(gt)
    # pred_pcd = read_obj_to_pcd(pred)
    # open3d.visualization.draw_geometries([pred_pcd])

    input_points, input_colors = read_obj(rgb)
    gt_points, gt_colors = read_obj(gt)
    method_points, method_colors = read_obj(pred)
    vis_multi_points([input_points, gt_points, method_points],
                     [input_colors, gt_colors, method_colors],
                     title=f'{name}-{idx}', plot_shape=(1, 3), point_size=12)
