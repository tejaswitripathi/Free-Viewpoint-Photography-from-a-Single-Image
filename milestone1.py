import os

# OpenCV needs this enabled before import to read .exr files
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import json
import subprocess
import tempfile

import cv2
import numpy as np
import open3d as o3d
import OpenEXR


S3_URI = "s3://tejas-blender-bucket/defocus-dataset/cafe/dataset/img_00000_f1.2_fl50.0_fd6.06/"


def download_s3_prefix(s3_uri, dest_dir):
    """Sync everything under an S3 prefix into a local directory."""
    subprocess.run(
        ["aws", "s3", "sync", s3_uri, dest_dir],
        check=True,
    )


def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def load_depth(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)

    if path.lower().endswith(".exr"):
        # Blender writes depth as a single, non-standard EXR channel (e.g.
        # "depth.V"), which OpenCV can't decode, so read it via OpenEXR.
        exr = OpenEXR.File(path)
        channels = exr.channels()
        channel = channels.get("depth.V") or next(iter(channels.values()))
        depth = np.asarray(channel.pixels, dtype=np.float32)
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # EXR depth can come in as multiple channels; keep a single channel
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    # Blender's Z pass uses a huge sentinel value for background/sky
    depth[depth >= 1e6] = 0.0

    # If depth was saved as 16-bit PNG in millimeters
    if depth.max() > 1000:
        depth = depth / 1000.0

    return depth


def get_intrinsics(metadata, width, height):
    focal_mm = metadata["focal_length_mm"]
    sensor_width_mm = metadata.get("sensor_width_mm", 36.0)

    fx = focal_mm / sensor_width_mm * width
    fy = fx

    cx = width / 2.0
    cy = height / 2.0

    return fx, fy, cx, cy


def rgbd_to_pointcloud(rgb, depth, fx, fy, cx, cy, depth_scale=1.0):
    h, w = depth.shape

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth * depth_scale

    valid = np.isfinite(z) & (z > 0)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Camera looks down +z (OpenCV convention); negate y and z to match
    # Open3D's view convention so near objects render in front.
    points = np.stack([x, -y, -z], axis=-1)
    colors = rgb

    points = points[valid]
    colors = colors[valid]

    return points, colors


def visualize(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])


def find_depth_file(data_dir):
    for name in os.listdir(data_dir):
        if name.lower().endswith(".exr"):
            return os.path.join(data_dir, name)
    raise FileNotFoundError(f"No .exr depth file found in {data_dir}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as data_dir:
        download_s3_prefix(S3_URI, data_dir)

        rgb = load_rgb(os.path.join(data_dir, "defocused.png"))
        depth = load_depth(find_depth_file(data_dir))

        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        h, w, _ = rgb.shape
        fx, fy, cx, cy = get_intrinsics(metadata, w, h)

        points, colors = rgbd_to_pointcloud(rgb, depth, fx, fy, cx, cy)

        print("points:", points.shape)
        print("depth range:", depth.min(), depth.max())

        visualize(points, colors)
