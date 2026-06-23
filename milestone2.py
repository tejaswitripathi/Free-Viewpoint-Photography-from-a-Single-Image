import os

# OpenCV needs this enabled before import to read .exr files
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import json
import subprocess
import tempfile
import math

import cv2
import numpy as np
import open3d as o3d
import OpenEXR


S3_URI = "s3://tejas-blender-bucket/defocus-dataset-multiview/bedroom/dataset/img_00000_f2.799999952316284_fl50.0_fd2.77/source/"


# -----------------------------
# Data loading
# -----------------------------

def download_s3_prefix(s3_uri, dest_dir):
    subprocess.run(["aws", "s3", "sync", s3_uri, dest_dir], check=True)


def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def load_depth(path):
    if path.endswith(".npy"):
        depth = np.load(path).astype(np.float32)

    elif path.lower().endswith(".exr"):
        exr = OpenEXR.File(path)
        channels = exr.channels()

        # Prefer Blender compositor depth channel
        channel = channels.get("depth.V") or next(iter(channels.values()))
        depth = np.asarray(channel.pixels, dtype=np.float32)

    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth image: {path}")
        depth = depth.astype(np.float32)

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    # Blender sky/background sentinel
    depth[depth >= 1e6] = 0.0

    # If 16-bit PNG in millimeters
    if depth.max() > 1000:
        depth = depth / 1000.0

    return depth


def find_depth_file(data_dir):
    for name in os.listdir(data_dir):
        if name.lower().endswith(".exr"):
            return os.path.join(data_dir, name)
    raise FileNotFoundError(f"No .exr depth file found in {data_dir}")


def get_intrinsics(metadata, width, height):
    focal_mm = metadata["focal_length_mm"]
    sensor_width_mm = metadata.get("sensor_width_mm", 36.0)

    fx = focal_mm / sensor_width_mm * width
    fy = fx

    cx = width / 2.0
    cy = height / 2.0

    return fx, fy, cx, cy


# -----------------------------
# Geometry
# -----------------------------

def rgbd_to_pointcloud(rgb, depth, fx, fy, cx, cy, stride=1):
    """
    Back-project RGB-D pixels into a colored 3D point cloud.

    stride=1 uses every pixel.
    stride=2 or 3 makes the viewer faster.
    """
    rgb = rgb[::stride, ::stride]
    depth = depth[::stride, ::stride]

    h, w = depth.shape

    # Important: intrinsics must be scaled when striding
    fx = fx / stride
    fy = fy / stride
    cx = cx / stride
    cy = cy / stride

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth
    valid = np.isfinite(z) & (z > 0)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # OpenCV camera convention: +z forward, +y down.
    # Open3D viewing feels nicer with y flipped and z negative.
    points = np.stack([x, -y, -z], axis=-1)
    colors = rgb

    points = points[valid]
    colors = colors[valid]

    return points, colors


def build_splat_pointcloud(points, colors, voxel_size=0.0, estimate_normals=True):
    """
    Milestone 2 approximation:
    Open3D point cloud with larger rendered points, optional voxel downsample,
    and normals for better interactive viewing.

    This is not true Gaussian rasterization yet.
    It is a depth-guided splat proxy.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    if estimate_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.25,
                max_nn=30,
            )
        )
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

    return pcd


# -----------------------------
# Interactive camera viewer
# -----------------------------

class CameraController:
    def __init__(self, vis):
        self.vis = vis

        self.move_step = 0.25
        self.rotate_step = 5.0

    def _get_params(self):
        ctr = self.vis.get_view_control()
        return ctr, ctr.convert_to_pinhole_camera_parameters()

    def _set_params(self, ctr, params):
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        self.vis.update_renderer()

    def translate_camera_local(self, dx=0.0, dy=0.0, dz=0.0):
        """
        Moves the camera in its local coordinate frame.
        dx: right
        dy: up
        dz: forward
        """
        ctr, params = self._get_params()

        extrinsic = params.extrinsic.copy()

        # Open3D extrinsic is world-to-camera.
        # Camera-to-world rotation is R.T.
        R_wc = extrinsic[:3, :3]
        R_cw = R_wc.T

        right = R_cw[:, 0]
        up = R_cw[:, 1]
        forward = R_cw[:, 2]

        delta_world = dx * right + dy * up + dz * forward

        # Since extrinsic is world-to-camera, translating camera in world
        # requires updating t = -R * C.
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]

        camera_center = -R.T @ t
        camera_center = camera_center + delta_world

        extrinsic[:3, 3] = -R @ camera_center
        params.extrinsic = extrinsic

        self._set_params(ctr, params)

    def rotate_view(self, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
        """
        Rotates camera around its current center.
        Good enough for Milestone 2 controls.
        """
        ctr = self.vis.get_view_control()

        # Open3D has built-in relative rotate in screen pixels.
        # The constants below are empirical.
        ctr.rotate(
            yaw_deg * 8.0,
            pitch_deg * 8.0,
        )
        self.vis.update_renderer()

    def register_callbacks(self):
        callbacks = {}

        # Movement
        callbacks[ord("W")] = lambda vis: self._cb(lambda: self.translate_camera_local(dz=self.move_step))
        callbacks[ord("S")] = lambda vis: self._cb(lambda: self.translate_camera_local(dz=-self.move_step))
        callbacks[ord("A")] = lambda vis: self._cb(lambda: self.translate_camera_local(dx=-self.move_step))
        callbacks[ord("D")] = lambda vis: self._cb(lambda: self.translate_camera_local(dx=self.move_step))
        callbacks[ord("R")] = lambda vis: self._cb(lambda: self.translate_camera_local(dy=self.move_step))
        callbacks[ord("F")] = lambda vis: self._cb(lambda: self.translate_camera_local(dy=-self.move_step))

        # Rotation
        callbacks[ord("J")] = lambda vis: self._cb(lambda: self.rotate_view(yaw_deg=-self.rotate_step))
        callbacks[ord("L")] = lambda vis: self._cb(lambda: self.rotate_view(yaw_deg=self.rotate_step))
        callbacks[ord("I")] = lambda vis: self._cb(lambda: self.rotate_view(pitch_deg=-self.rotate_step))
        callbacks[ord("K")] = lambda vis: self._cb(lambda: self.rotate_view(pitch_deg=self.rotate_step))

        # Zoom-ish movement
        callbacks[ord("Q")] = lambda vis: self._cb(lambda: self.translate_camera_local(dz=-2 * self.move_step))
        callbacks[ord("E")] = lambda vis: self._cb(lambda: self.translate_camera_local(dz=2 * self.move_step))

        return callbacks

    def _cb(self, fn):
        fn()
        return False


def visualize_interactive(pcd, point_size=4.0):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Milestone 2: Depth-Guided Splat Viewer", width=1400, height=900)

    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1.0, 1.0, 1.0])
    render_opt.point_size = point_size
    render_opt.show_coordinate_frame = True

    controller = CameraController(vis)
    for key, cb in controller.register_callbacks().items():
        vis.register_key_callback(key, cb)

    print_controls()

    vis.run()
    vis.destroy_window()


def print_controls():
    print()
    print("Controls:")
    print("  W / S : move forward / backward")
    print("  A / D : move left / right")
    print("  R / F : move up / down")
    print("  I / K : look up / down")
    print("  J / L : look left / right")
    print("  Q / E : larger backward / forward step")
    print()

def points_to_gaussians(points, colors):
    N = len(points)

    d = np.linalg.norm(points, axis=1)

    s = 0.01 * d

    scales = np.stack([s,s,s], axis=1)

    opacity = np.ones((N, 1), dtype=np.float32)

    rotation = np.zeros((N, 4), dtype=np.float32)
    rotation[:, 0] = 1.0  # identity quaternion

    return {
        "position": points,
        "color": colors,
        "scale": scales,
        "opacity": opacity,
        "rotation": rotation,
    }


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as data_dir:
        print("Downloading sample...")
        download_s3_prefix(S3_URI, data_dir)

        rgb = load_rgb(os.path.join(data_dir, "_compositor_dummy.png"))
        depth = load_depth(find_depth_file(data_dir))

        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        h, w, _ = rgb.shape
        fx, fy, cx, cy = get_intrinsics(metadata, w, h)

        points, colors = rgbd_to_pointcloud(
            rgb,
            depth,
            fx,
            fy,
            cx,
            cy,
            stride=1,
        )

        print("points:", points.shape)
        print("depth range:", float(depth[depth > 0].min()), float(depth.max()))

        pcd = build_splat_pointcloud(
            points,
            colors,
            voxel_size=0.0,
            estimate_normals=True,
        )

        gaussians = points_to_gaussians(points, colors)

        print("Gaussians:", len(gaussians["position"]))
        print("Mean scale:", gaussians["scale"].mean())

        np.savez(
            "sample.npz",
            **gaussians,
        )

        visualize_interactive(
            pcd,
            point_size=4.0,
        )