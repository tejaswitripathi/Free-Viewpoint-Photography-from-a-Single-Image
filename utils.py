import os

# OpenCV needs this enabled before import to read .exr files
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import OpenEXR

def download_s3_prefix(s3_uri: str, dest_dir: str) -> None:
    subprocess.run(["aws", "s3", "sync", s3_uri, dest_dir], check=True)


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def load_depth(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        depth = np.load(path).astype(np.float32)

    elif path.lower().endswith(".exr"):
        # Blender compositor output often stores depth in a named EXR channel
        # such as "depth.V", which cv2.imread may not decode correctly.
        exr = OpenEXR.File(path)
        channels = exr.channels()

        channel = channels.get("depth.V")
        if channel is None:
            # Fall back to the first channel if your file uses a different name.
            channel = next(iter(channels.values()))

        depth = np.asarray(channel.pixels, dtype=np.float32)

    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth image: {path}")
        depth = depth.astype(np.float32)

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    # Blender's sky/background Z-pass sentinel.
    depth[depth >= 1e6] = 0.0

    # If depth was saved as 16-bit PNG in millimeters.
    if depth.size > 0 and depth.max() > 1000:
        depth = depth / 1000.0

    return depth


def find_depth_file(data_dir: str) -> str:
    for name in os.listdir(data_dir):
        if name.lower().endswith(".exr"):
            return os.path.join(data_dir, name)
    for name in os.listdir(data_dir):
        if name.lower().endswith(".npy"):
            return os.path.join(data_dir, name)
    raise FileNotFoundError(f"No .exr or .npy depth file found in {data_dir}")


def find_rgb_file(data_dir: str) -> str:
    """
    Supports both your old defocus dataset layout and the new multiview layout.
    Priority:
      1. rgb.png          new multiview source
      2. sharp.png        old refocus dataset
      3. defocused.png    old refocus dataset
      4. _compositor_dummy.png as last resort
    """
    candidates = ["rgb.png", "sharp.png", "defocused.png", "_compositor_dummy.png"]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No RGB image found in {data_dir}. Expected one of {candidates}"
    )


def get_intrinsics(metadata: dict, width: int, height: int):
    focal_mm = metadata["focal_length_mm"]
    sensor_width_mm = metadata.get("sensor_width_mm", 36.0)

    fx = focal_mm / sensor_width_mm * width

    # If you later store sensor height, use it. Otherwise assume square pixels.
    sensor_height_mm = metadata.get("sensor_height_mm")
    if sensor_height_mm is not None:
        fy = focal_mm / sensor_height_mm * height
    else:
        fy = fx

    cx = width / 2.0
    cy = height / 2.0

    return fx, fy, cx, cy


# -----------------------------
# Geometry
# -----------------------------

def rgbd_to_pointcloud(rgb, depth, fx, fy, cx, cy, stride=1):
    """
    Back-project RGB-D pixels into camera-space 3D points.

    Convention used in this file:
      point = [x, -y, -z]

    This matches your Open3D-friendly convention from Milestone 1/2.
    Projection functions below convert it back to camera-forward +z.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")

    rgb_s = rgb[::stride, ::stride]
    depth_s = depth[::stride, ::stride]

    h, w = depth_s.shape

    fx_s = fx / stride
    fy_s = fy / stride
    cx_s = cx / stride
    cy_s = cy / stride

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth_s
    valid = np.isfinite(z) & (z > 0)

    x = (u - cx_s) * z / fx_s
    y = (v - cy_s) * z / fy_s

    points = np.stack([x, -y, -z], axis=-1)
    colors = rgb_s

    return points[valid], colors[valid]


def points_to_gaussians(points, colors, fx, base_world_scale=0.004,
                        min_radius_px=1.5, max_radius_px=7.0,
                        opacity=0.55):
    """
    Milestone 3 Gaussian representation.

    This is still the simple version:
      - position comes from depth backprojection
      - color comes from RGB
      - scale becomes an isotropic projected radius in pixels
      - opacity is constant
      - rotation is identity quaternion

    Later, a neural network can replace radius/opacity/rotation.
    """
    n = len(points)

    radii_px = compute_depth_based_radii(
        points,
        fx=fx,
        base_world_scale=base_world_scale,
        min_px=min_radius_px,
        max_px=max_radius_px,
    )

    opacities = np.full((n,), opacity, dtype=np.float32)

    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # identity quaternion: w, x, y, z

    return {
        "position": points.astype(np.float32),
        "color": colors.astype(np.float32),
        "radius_px": radii_px.astype(np.float32),
        "opacity": opacities.astype(np.float32),
        "rotation_quat_wxyz": rotations,
    }


def project_points(points, fx, fy, cx, cy):
    """
    Project 3D points into pixel coordinates.

    Input points use your stored convention:
      [x, -y, -z]

    Internally we convert back to:
      x = x
      y = -stored_y
      z = -stored_z
    """
    x = points[:, 0]
    y = -points[:, 1]
    z = -points[:, 2]

    valid = z > 1e-6

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    return u, v, z, valid


def compute_depth_based_radii(points, fx, base_world_scale=0.004,
                              min_px=1.5, max_px=7.0):
    """
    Convert a world-space Gaussian size into a projected 2D splat radius.

    world_radius = base_world_scale * z

    Because this first version scales world radius with depth, the projected
    radius is intentionally fairly stable across depth:
      radius_px ~= fx * base_world_scale

    You can later make this learned or surface-adaptive.
    """
    z = -points[:, 2]
    z = np.maximum(z, 1e-6)

    world_radius = base_world_scale * z
    radius_px = fx * world_radius / z

    return np.clip(radius_px, min_px, max_px).astype(np.float32)


# -----------------------------
# Camera transforms for novel views
# -----------------------------

def rotation_matrix_yaw_pitch_roll(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """
    Camera-space rotation matrix.

    Positive yaw turns the camera to the right.
    Positive pitch turns the camera upward.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # Rotations in camera-forward +z convention.
    R_yaw = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ], dtype=np.float32)

    R_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cp, -sp],
        [0.0, sp, cp],
    ], dtype=np.float32)

    R_roll = np.array([
        [cr, -sr, 0.0],
        [sr, cr, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return R_roll @ R_pitch @ R_yaw


def transform_points_camera(points, tx=0.0, ty=0.0, tz=0.0,
                            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """
    Render the same scene from a new local camera pose.

    tx, ty, tz represent camera motion in meters:
      tx > 0: camera moves right
      ty > 0: camera moves up
      tz > 0: camera moves forward

    yaw/pitch/roll represent camera rotation.

    We transform world/source-camera points into the new camera coordinate frame.
    """
    # Convert stored [x, -y, -z] back to camera-forward coordinates.
    p = np.empty_like(points, dtype=np.float32)
    p[:, 0] = points[:, 0]
    p[:, 1] = -points[:, 1]
    p[:, 2] = -points[:, 2]

    # Moving the camera by t is equivalent to subtracting t from scene points.
    t = np.array([tx, ty, tz], dtype=np.float32)
    p = p - t[None, :]

    # Camera rotation: world-to-new-camera uses inverse rotation.
    R = rotation_matrix_yaw_pitch_roll(yaw_deg, pitch_deg, roll_deg)
    p = (R.T @ p.T).T

    # Convert back to stored convention.
    out = np.empty_like(points, dtype=np.float32)
    out[:, 0] = p[:, 0]
    out[:, 1] = -p[:, 1]
    out[:, 2] = -p[:, 2]
    return out


# -----------------------------
# CPU Gaussian splat renderer
# -----------------------------

def render_gaussian_splats(points, colors, fx, fy, cx, cy, width, height,
                           radii_px=None, opacities=None,
                           background=(1.0, 1.0, 1.0),
                           max_points=None):
    """
    Simple CPU Gaussian splat renderer.

    This proves the Milestone 3 math:
      3D Gaussian centers -> 2D Gaussian splats -> alpha-composited image.

    Notes:
      - It uses circular 2D Gaussians, not full anisotropic ellipses yet.
      - It is intentionally simple and slow.
      - For speed, use stride=2/3 or max_points.
    """
    image = np.zeros((height, width, 3), dtype=np.float32)
    transmittance = np.ones((height, width), dtype=np.float32)

    u, v, z, valid = project_points(points, fx, fy, cx, cy)

    idxs = np.where(valid)[0]

    # Only splats whose centers are near the image are worth considering.
    margin = 64
    in_bounds = (
        (u[idxs] >= -margin) & (u[idxs] < width + margin) &
        (v[idxs] >= -margin) & (v[idxs] < height + margin)
    )
    idxs = idxs[in_bounds]

    if max_points is not None and len(idxs) > max_points:
        # Keep deterministic subset spread across all valid depths.
        rng = np.random.default_rng(42)
        idxs = rng.choice(idxs, size=max_points, replace=False)

    # Front-to-back compositing with transmittance:
    # C += T * alpha * color
    # T *= (1 - alpha)
    idxs = idxs[np.argsort(z[idxs])]

    if radii_px is None:
        radii_px = np.full((len(points),), 2.5, dtype=np.float32)

    if opacities is None:
        opacities = np.full((len(points),), 0.55, dtype=np.float32)

    for count, idx in enumerate(idxs):
        ui = float(u[idx])
        vi = float(v[idx])

        radius = float(radii_px[idx])
        if radius <= 0:
            continue

        sigma = max(radius / 2.0, 1e-3)

        xmin = max(0, int(math.floor(ui - 3 * sigma)))
        xmax = min(width - 1, int(math.ceil(ui + 3 * sigma)))
        ymin = max(0, int(math.floor(vi - 3 * sigma)))
        ymax = min(height - 1, int(math.ceil(vi + 3 * sigma)))

        if xmin >= xmax or ymin >= ymax:
            continue

        xs = np.arange(xmin, xmax + 1, dtype=np.float32)
        ys = np.arange(ymin, ymax + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)

        d2 = (xx - ui) ** 2 + (yy - vi) ** 2
        weight = np.exp(-0.5 * d2 / (sigma ** 2)).astype(np.float32)

        alpha = np.clip(float(opacities[idx]) * weight, 0.0, 0.995)

        patch_img = image[ymin:ymax + 1, xmin:xmax + 1]
        patch_T = transmittance[ymin:ymax + 1, xmin:xmax + 1]

        contribution = patch_T[..., None] * alpha[..., None] * colors[idx][None, None, :]
        patch_img += contribution
        patch_T *= (1.0 - alpha)

    bg = np.array(background, dtype=np.float32)
    image += transmittance[..., None] * bg[None, None, :]

    return np.clip(image, 0.0, 1.0)


def save_rgb(path, image):
    path = str(path)
    bgr = cv2.cvtColor((np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def psnr(pred, target):
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)