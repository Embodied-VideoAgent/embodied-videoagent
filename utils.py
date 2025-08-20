import math
import cv2
import numpy as np
from object3d import Object3D


def camera3d_to_world3d_transformation(xyz_local, position, rmat):
    rmat = rmat.T
    xyz_world = xyz_local @ rmat + position
    return xyz_world


def world3d_to_camera3d_transformation(xyz_world, position, rmat):
    rmat = rmat.T
    invert_rmat = np.linalg.inv(rmat)
    xyz_camera = (xyz_world - position) @ invert_rmat
    return xyz_camera


def depth2d_to_frame2d(depth):
    h, w = depth.shape
    i, j = np.indices((h, w))
    xyz_frame = np.stack([i, j, depth], axis=2)
    return xyz_frame.reshape(-1, 3)


def frame2d_to_camera3d_transformation(xyz_frame, h, w, hfov):
    fx = (0.5 * w) / math.tan(math.radians(0.5 * hfov))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    i = xyz_frame[:, 0]
    j = xyz_frame[:, 1]
    z = xyz_frame[:, 2]

    x = (j - cx) * z / fx
    y = (i - cy) * z / fy

    xyz_local = np.stack((x, y, z), axis=-1)
    return xyz_local


def camera3d_to_frame2d_transformation(xyz_local, h, w, hfov, epsilon=1e-5):
    fx = (0.5 * w) / math.tan(math.radians(0.5 * hfov))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    x = xyz_local[:, 0]
    y = xyz_local[:, 1]
    z = xyz_local[:, 2]
    sign_z = np.sign(z)
    sign_z[sign_z == 0] = 1
    z_safe = np.where(np.abs(z) < epsilon, sign_z * epsilon, z)
    i = cy + y * fy / np.abs(z_safe)
    j = cx + x * fx / np.abs(z_safe)
    xyz_frame = np.stack([i, j, z_safe], axis=-1)
    return xyz_frame


def depth2d_to_world3d_transformation(depth, position, rmat, hfov):
    h, w = depth.shape
    xyz_frame = depth2d_to_frame2d(depth)
    xyz_local = frame2d_to_camera3d_transformation(
        xyz_frame=xyz_frame, h=h, w=w, hfov=hfov
    )
    xyz_world = camera3d_to_world3d_transformation(
        xyz_local=xyz_local, position=position, rmat=rmat
    )
    xyz_world = xyz_world.reshape(h, w, 3)
    return xyz_world


def world3d_to_frame2d_transformation(xyz_world, position, h, w, rmat, hfov):
    xyz_local = world3d_to_camera3d_transformation(
        xyz_world=xyz_world, position=position, rmat=rmat
    )
    xyz_frame = camera3d_to_frame2d_transformation(
        xyz_local=xyz_local, h=h, w=w, hfov=hfov
    )
    return xyz_frame


def plot_line_on_frame(xyz_frame_0, xyz_frame_1, bgr):
    """Given two points in frame, plot their line on the frame if visible."""
    py0, px0, z0 = xyz_frame_0
    py1, px1, z1 = xyz_frame_1
    if z0 < 0 and z1 < 0:  # both points are behind the camera plane
        return
    color = (0, 255, 0)
    thickness = 1
    cv2.line(bgr, (int(px0), int(py0)), (int(px1), int(py1)), color, thickness)


def plot_object_bbox_on_frame(obj: Object3D, bgr, pos, rmat, hfov):
    """Plot object 3D bbox on the frame."""
    h, w = bgr.shape[:2]
    x0, y0, z0 = obj.min_xyz
    x1, y1, z1 = obj.max_xyz
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y0, z1],
            [x0, y0, z1],
            [x0, y1, z0],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y1, z1],
            [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2],
        ]
    )
    xyz_local = world3d_to_camera3d_transformation(
        xyz_world=vertices, position=pos, rmat=rmat
    )
    xyz_frame = camera3d_to_frame2d_transformation(
        xyz_local=xyz_local, h=h, w=w, hfov=hfov
    )
    a0, b0, c0, d0, a1, b1, c1, d1, center = xyz_frame
    plot_line_on_frame(a0, b0, bgr)
    plot_line_on_frame(b0, c0, bgr)
    plot_line_on_frame(c0, d0, bgr)
    plot_line_on_frame(d0, a0, bgr)
    plot_line_on_frame(a1, b1, bgr)
    plot_line_on_frame(b1, c1, bgr)
    plot_line_on_frame(c1, d1, bgr)
    plot_line_on_frame(d1, a1, bgr)
    plot_line_on_frame(a0, a1, bgr)
    plot_line_on_frame(b0, b1, bgr)
    plot_line_on_frame(c0, c1, bgr)
    plot_line_on_frame(d0, d1, bgr)
    if 0 <= center[0] < h and 0 <= center[1] < w and center[2] > 0:
        color = (255, 255, 0)
        cv2.putText(
            bgr,
            f"{obj.identifier}:{obj.category}",
            (int(center[1]), int(center[0])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )
    return bgr


def plot_object_id_on_frame(obj: Object3D, bgr, pos, rmat, hfov):
    """Plot object ID on the frame."""
    h, w = bgr.shape[:2]
    x0, y0, z0 = obj.min_xyz
    x1, y1, z1 = obj.max_xyz
    center = np.array([[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]])
    xyz_local = world3d_to_camera3d_transformation(
        xyz_world=center, position=pos, rmat=rmat
    )
    xyz_frame = camera3d_to_frame2d_transformation(
        xyz_local=xyz_local, h=h, w=w, hfov=hfov
    )
    center = xyz_frame[0]
    py, px, d = int(center[0]), int(center[1]), center[2]
    if 0 <= py < h and 0 <= px < w and d > 0:
        cv2.circle(bgr, (px, py), 12, (0, 0, 160), -1)
        cv2.putText(
            bgr,
            str(obj.identifier),
            (px - 8, py + 2),
            cv2.FONT_HERSHEY_PLAIN,
            1.2,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return bgr


def check_visible_and_crop(obj: Object3D, pos, rmat, hfov, depth):
    h, w = depth.shape
    x0, y0, z0 = obj.min_xyz
    x1, y1, z1 = obj.max_xyz
    #center = np.array([[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]])
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y0, z1],
            [x0, y0, z1],
            [x0, y1, z0],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y1, z1],
            [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2],
        ]
    )
    xyz_local = world3d_to_camera3d_transformation(
        xyz_world=vertices, position=pos, rmat=rmat
    )
    xyz_frame = camera3d_to_frame2d_transformation(
        xyz_local=xyz_local, h=h, w=w, hfov=hfov
    )

    max_y = int(np.max(xyz_frame[:, 0], axis=0))
    min_y = int(np.min(xyz_frame[:, 0], axis=0))
    max_x = int(np.max(xyz_frame[:, 1], axis=0))
    min_x = int(np.min(xyz_frame[:, 1], axis=0))
    
    if max_y >= h or min_y < 0 or max_x >= w or min_x < 0:
        return False, [0, 0, 0, 0]
    
    py, px, _ = xyz_frame[-1]
    py, px = int(py), int(px)
    
    if not (0 <= py < h and 0 <= px < w):
        return False, [0, 0, 0, 0]
    center_pixel = np.array([[py, px, depth[py, px]]])
    xyz_local = frame2d_to_camera3d_transformation(
        xyz_frame=center_pixel, h=h, w=w, hfov=hfov
    )
    xyz_world = camera3d_to_world3d_transformation(
        xyz_local=xyz_local, position=pos, rmat=rmat
    )
    center_pixel_world = xyz_world[0]
    if np.all(center_pixel_world >= obj.min_xyz) and np.all(
        center_pixel_world <= obj.max_xyz):
        return True, [min_y, max_y, min_x, max_x]
    else:
        return False, [0, 0, 0, 0]


def check_dynamic(obj: Object3D, pos, rmat, hfov, depth):
    h, w = depth.shape
    x0, y0, z0 = obj.min_xyz
    x1, y1, z1 = obj.max_xyz
    center = np.array([[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]])
    xyz_local = world3d_to_camera3d_transformation(
        xyz_world=center, position=pos, rmat=rmat
    )
    xyz_frame = camera3d_to_frame2d_transformation(
        xyz_local=xyz_local, h=h, w=w, hfov=hfov
    )
    py, px, estimate_depth = xyz_frame[0]
    py, px = int(py), int(px)
    if not (0 <= py < h and 0 <= px < w):
        return False
    center_pixel = np.array([[py, px, depth[py, px]]])
    xyz_local = frame2d_to_camera3d_transformation(
        xyz_frame=center_pixel, h=h, w=w, hfov=hfov
    )
    xyz_world = camera3d_to_world3d_transformation(
        xyz_local=xyz_local, position=pos, rmat=rmat
    )
    center_pixel_world = xyz_world[0]
    inside_bbox = np.all(center_pixel_world >= obj.min_xyz) and np.all(
        center_pixel_world <= obj.max_xyz
    )
    return depth[py, px] > estimate_depth and not inside_bbox




def plot_visible_object_ids(object_list: list[Object3D], bgr, pos, rmat, hfov, depth):
    for obj in object_list:
        visible, bbox =  check_visible_and_crop(obj, pos, rmat, hfov, depth)
        if visible:
            plot_object_id_on_frame(obj, bgr, pos, rmat, hfov)
    return bgr


def plot_visible_object_bboxes(object_list: list[Object3D], bgr, pos, rmat, hfov, depth):
    for obj in object_list:
        visible, bbox =  check_visible_and_crop(obj, pos, rmat, hfov, depth)
        if visible:
            plot_object_bbox_on_frame(obj, bgr, pos, rmat, hfov)
    return bgr