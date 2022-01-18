import numpy as np
from scipy import stats

def rotdiff(x, y):
    """calc. the angular difference between x and y in [-pi, pi]"""
    d = (x - y) % (2 * np.pi)
    return np.where(d > np.pi, d - 2 * np.pi, d)

def regression_precision(y_true, y_pred, variance_pred, confidence):
    r"""(Kuleshov et al., 2018)
    Predicted value has an uncertainty. Given confidence p,
    the estimated region is derived inside which y_true lies with proba. p.
    The estimated region is equal to the p-confidence region.

    Parameters:
        y_true: float ndarray with shape (n_sample,)
        y_pred: float ndarray with shape (n_sample,)
        variance_pred: float ndarray with shape (n_sample,)
            Estimated variance \sigma^2 for each predicted y.
        confidence: float
            Threshold to count correct predctions.
    
    Returns:
        precision: float
    """
    sigma_pred = np.sqrt(variance_pred)
    region_lowers, region_uppers = stats.norm.interval(confidence, loc=y_pred, scale=sigma_pred)
    correct_count = np.sum((region_lowers <= y_true) & (y_true <= region_uppers))
    return correct_count / len(y_true)

def regression_precisionNd(y_true, y_pred, variance_pred, confidence, df):
    r"""
    ND Gaussian version of regression_precision.

    Parameters:
        y_true: float ndarray with shape (n_sample, df)
        y_pred: float ndarray with shape (n_sample, df)
        variance_pred: float ndarray with shape (n_sample, df, df)
            Estimated covariance matrix \Sigma for each predicted y.
        confidence: float
            Threshold to count correct predctions.
        df: degree of multivariate Gaussian
    
    Returns:
        precision: float
    """
    region_sq_mahalanobis = stats.chi2.ppf(confidence, df=df)
    y_diff = y_true - y_pred
    diff_sq_mahalanobis = (y_diff[:,None,:] @ np.linalg.inv(variance_pred) @ y_diff[:,:,None]).ravel()
    correct_count = np.sum(diff_sq_mahalanobis <= region_sq_mahalanobis)
    return correct_count / len(y_true)

def regression_precisionAngular(y_true, y_pred, variance_pred, confidence):
    """
    Angular version of regression_precision.
    This assumes the Gaussian distribution, which is the result of approximating von-Mises distribution.
    Parameters:
        y_true: float ndarray with shape (n_sample,)
        y_pred: float ndarray with shape (n_sample,)
        variance_pred: float ndarray with shape (n_sample,)
            Estimated variance \sigma^2 for each predicted y.
        confidence: float
            Threshold to count correct predctions.
    
    Returns:
        precision: float
    """
    region_sq_mahalanobis = stats.chi2.ppf(confidence, df=1)
    y_diff = rotdiff(y_true, y_pred)
    diff_sq_mahalanobis = y_diff ** 2 / variance_pred
    correct_count = np.sum(diff_sq_mahalanobis <= region_sq_mahalanobis)
    return correct_count / len(y_true)

def reg_calibration_curve(y_true, y_pred, variance_pred, n_bins):
    confidences = np.linspace(0.0, 1.0, n_bins+1, endpoint=True)
    precs = np.zeros(n_bins+1, dtype=np.float32)
    counts = np.zeros(n_bins+1, dtype=np.float32)
    for i, conf in enumerate(confidences):
        precs[i] = regression_precision(y_true, y_pred, variance_pred, conf)
        if i > 0:
            counts[i] = precs[i] * len(y_true) - precs[i-1] * len(y_true)
    return precs[1:], confidences[1:], counts[1:]

def reg_calibration_curve_nd(y_true, y_pred, covariance_pred, n_bins):
    confidences = np.linspace(0.0, 1.0, n_bins+1, endpoint=True)
    precs = np.zeros(n_bins+1, dtype=np.float32)
    counts = np.zeros(n_bins+1, dtype=np.float32)
    for i, conf in enumerate(confidences):
        precs[i] = regression_precisionNd(y_true, y_pred, covariance_pred, conf, df=y_true.shape[1])
        if i > 0:
            counts[i] = precs[i] * len(y_true) - precs[i-1] * len(y_true)
    return precs[1:], confidences[1:], counts[1:]

def reg_calibration_curve_angular(y_true, y_pred, variance_pred, n_bins):
    confidences = np.linspace(0.0, 1.0, n_bins+1, endpoint=True)
    precs = np.zeros(n_bins+1, dtype=np.float32)
    counts = np.zeros(n_bins+1, dtype=np.float32)
    for i, conf in enumerate(confidences):
        precs[i] = regression_precisionAngular(y_true, y_pred, variance_pred, conf)
        if i > 0:
            counts[i] = precs[i] * len(y_true) - precs[i-1] * len(y_true)
    return precs[1:], confidences[1:], counts[1:]

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point. 
    
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
    
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners

def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_T)

def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=(0.5, 0.5, 0.5),
                           axis=2):
    """convert kitti locations, dimensions and angles to corners
    
    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners

def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners. 
    Returns:
        surfaces (float array, [N, 6, 4, 3]): 
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces

def corner_to_surfaces_3d_jit(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners. 
    Returns:
        surfaces (float array, [N, 6, 4, 3]): 
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array([
        0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7
    ]).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces

def surface_equ_3d_jitv2(surfaces):
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    num_polygon = surfaces.shape[0]
    max_num_surfaces = surfaces.shape[1]
    normal_vec = np.zeros((num_polygon, max_num_surfaces, 3), dtype=surfaces.dtype)
    d = np.zeros((num_polygon, max_num_surfaces), dtype=surfaces.dtype)
    sv0 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    sv1 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    for i in range(num_polygon):
        for j in range(max_num_surfaces):
            sv0[0] = surfaces[i, j, 0, 0] - surfaces[i, j, 1, 0]
            sv0[1] = surfaces[i, j, 0, 1] - surfaces[i, j, 1, 1]
            sv0[2] = surfaces[i, j, 0, 2] - surfaces[i, j, 1, 2]
            sv1[0] = surfaces[i, j, 1, 0] - surfaces[i, j, 2, 0]
            sv1[1] = surfaces[i, j, 1, 1] - surfaces[i, j, 2, 1]
            sv1[2] = surfaces[i, j, 1, 2] - surfaces[i, j, 2, 2]
            normal_vec[i, j, 0] = (sv0[1] * sv1[2] - sv0[2] * sv1[1])
            normal_vec[i, j, 1] = (sv0[2] * sv1[0] - sv0[0] * sv1[2])
            normal_vec[i, j, 2] = (sv0[0] * sv1[1] - sv0[1] * sv1[0])
            
            d[i, j] = -surfaces[i, j, 0, 0] * normal_vec[i, j, 0] - \
                      surfaces[i, j, 0, 1] * normal_vec[i, j, 1] - \
                       surfaces[i, j, 0, 2] * normal_vec[i, j, 2]
    return normal_vec, d

def _points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    normal_vec, d,
                                    num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces, 
            max_num_points_of_surface, 3] 
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces 
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = points[i, 0] * normal_vec[j, k, 0] \
                     + points[i, 1] * normal_vec[j, k, 1] \
                     + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces, 
            max_num_points_of_surface, 3] 
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces 
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d, num_surfaces)

def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0.5)):
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=z_axis)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices

def camera_to_lidar(points, r_rect, velo2cam):
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    points_shape = list(points.shape[:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = points @ (r_rect @ velo2cam).T
    return camera_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    xyz = data[:, 0:3]
    l, h, w = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)


def box_lidar_to_camera(data, r_rect, velo2cam):
    xyz_lidar = data[:, 0:3]
    w, l, h = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return np.concatenate([xyz, l, h, w, r], axis=1)

def change_box3d_center_(box3d, src, dst):
    dst = np.array(dst, dtype=box3d.dtype)
    src = np.array(src, dtype=box3d.dtype)
    box3d[..., :3] += box3d[..., 3:6] * (dst - src)