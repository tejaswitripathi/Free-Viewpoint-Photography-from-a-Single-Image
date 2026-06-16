import bpy
import os
import json
import random
import math
import shutil
import subprocess
import numpy as np
from mathutils import Vector

# ----------------------------
# CONFIG
# ----------------------------

random.seed(42)

S3_BUCKET = "tejas-blender-bucket"
S3_PREFIX = "defocus-dataset-multiview"
scene_name = os.path.basename(os.path.dirname(bpy.data.filepath)) or "unknown_scene"

# For multiview, start smaller. Pose variation matters more than aperture variation.
# focus_distances = np.linspace(4, 28, 10).tolist()

datadir = "//dataset/"
sensor_width_mm = 35.0

resolution_x = 512
resolution_y = 512
samples = 64

camera = bpy.context.scene.camera
scene = bpy.context.scene

# Use the aperture and focal length already configured on the .blend camera.
f_stop = camera.data.dof.aperture_fstop
focal_length = camera.data.lens

# ----------------------------
# BASIC RENDER SETTINGS
# ----------------------------

scene.render.engine = "CYCLES"
scene.cycles.samples = samples
scene.cycles.use_denoising = True

scene.render.resolution_x = resolution_x
scene.render.resolution_y = resolution_y
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"

camera.data.sensor_width = sensor_width_mm

# ----------------------------
# HELPERS
# ----------------------------

def ensure_dir(path):
    os.makedirs(bpy.path.abspath(path), exist_ok=True)


def upload_and_cleanup(folder):
    local = bpy.path.abspath(folder)
    sample = os.path.basename(os.path.normpath(local))
    s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/{scene_name}/dataset/{sample}/"

    try:
        subprocess.run(["aws", "s3", "sync", local, s3_uri], check=True)
        shutil.rmtree(local, ignore_errors=True)
        print(f"Uploaded and removed {local} -> {s3_uri}")
    except Exception as e:
        print(f"Upload failed for {local}, keeping local copy: {e}")


def object_world_center(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return sum(corners, Vector()) / 8


def dist_to_camera(obj):
    return (object_world_center(obj) - camera.location).length


def get_candidate_objects(start, end):
    candidates = []

    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        if obj.hide_render:
            continue

        d = dist_to_camera(obj)

        if start <= d < end:
            candidates.append(obj)

    return candidates


def make_adaptive_depth_bins(num_bins=10):
    distances = []

    for obj in scene.objects:
        if obj.type == "MESH" and not obj.hide_render:
            distances.append(dist_to_camera(obj))

    distances = sorted([d for d in distances if d > 0])

    if not distances:
        raise RuntimeError("No visible mesh objects found.")

    min_d = max(0.1, distances[0])
    max_d = distances[-1]

    edges = np.linspace(min_d, max_d, num_bins + 1).tolist()

    return list(zip(edges[:-1], edges[1:]))


def assign_object_indices():
    idx = 1
    mapping = {}

    for obj in scene.objects:
        if obj.type == "MESH":
            obj.pass_index = idx
            mapping[obj.name] = idx
            idx += 1

    return mapping


def render_png(filepath):
    scene.render.filepath = filepath
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)


def setup_depth_nodes(output_dir):
    if hasattr(scene, "compositing_node_group"):
        for ng in list(bpy.data.node_groups):
            if ng.name.startswith("DepthCompositor"):
                bpy.data.node_groups.remove(ng)

        tree = bpy.data.node_groups.new("DepthCompositor", "CompositorNodeTree")
        scene.compositing_node_group = tree

        render_layers = tree.nodes.new(type="CompositorNodeRLayers")

        depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_output.directory = bpy.path.abspath(output_dir)
        depth_output.file_name = "depth_"
        depth_output.format.file_format = "OPEN_EXR_MULTILAYER"
        depth_output.format.color_depth = "32"

        depth_output.file_output_items.new("FLOAT", "depth")
        tree.links.new(render_layers.outputs["Depth"], depth_output.inputs["depth"])

    else:
        scene.use_nodes = True
        scene.render.use_compositing = True

        tree = scene.node_tree
        tree.nodes.clear()

        render_layers = tree.nodes.new(type="CompositorNodeRLayers")

        depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_output.base_path = bpy.path.abspath(os.path.join(output_dir, "depth_"))
        depth_output.format.file_format = "OPEN_EXR_MULTILAYER"
        depth_output.format.color_depth = "32"

        depth_output.layer_slots.clear()
        depth_output.layer_slots.new("depth")
        tree.links.new(render_layers.outputs["Depth"], depth_output.inputs["depth"])


def disable_nodes():
    if hasattr(scene, "compositing_node_group"):
        scene.compositing_node_group = None
    elif hasattr(scene, "use_nodes"):
        scene.use_nodes = False


# ----------------------------
# CAMERA POSE HELPERS
# ----------------------------

def camera_pose_dict(cam):
    return {
        "location": list(cam.location),
        "rotation_euler": list(cam.rotation_euler),
        "lens_mm": cam.data.lens,
        "sensor_width_mm": cam.data.sensor_width,
        "shift_x": cam.data.shift_x,
        "shift_y": cam.data.shift_y,
    }


def restore_camera_pose(cam, pose):
    cam.location = Vector(pose["location"])
    cam.rotation_euler = pose["rotation_euler"]
    cam.data.lens = pose["lens_mm"]
    cam.data.sensor_width = pose["sensor_width_mm"]
    cam.data.shift_x = pose["shift_x"]
    cam.data.shift_y = pose["shift_y"]


def look_at(cam, target):
    direction = target - cam.location
    quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = quat.to_euler()


def get_camera_basis(cam):
    q = cam.matrix_world.to_quaternion()
    right = q @ Vector((1, 0, 0))
    up = q @ Vector((0, 1, 0))
    forward = q @ Vector((0, 0, -1))
    return right, up, forward


def move_camera_local(cam, dx=0.0, dy=0.0, dz=0.0):
    right, up, forward = get_camera_basis(cam)
    cam.location = cam.location + dx * right + dy * up + dz * forward


def scene_ray_cast(origin, direction, max_distance):
    """
    Cast a ray through the evaluated scene geometry.

    Returns (hit, location, obj). `location` and `obj` are only meaningful
    when `hit` is True.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()

    length = direction.length
    if length < 1e-9 or max_distance <= 0.0:
        return False, None, None

    hit, location, normal, index, obj, matrix = scene.ray_cast(
        depsgraph,
        origin,
        direction / length,
        distance=max_distance,
    )

    return hit, location, obj


def collision_safe_offset(cam, source_pose, offset, margin=0.25):
    """
    Prevent the camera from translating through walls, chairs, tables, etc.

    The translational part of the offset (dx/dy/dz, in camera-local space) is
    cast as a ray from the source pose. If geometry is hit before the camera
    reaches its target position, the translation is scaled back so the camera
    stops `margin` meters short of the obstacle. Rotational components
    (yaw/pitch) are left untouched.

    Returns a new offset dict with possibly reduced dx/dy/dz.
    """
    restore_camera_pose(cam, source_pose)
    right, up, forward = get_camera_basis(cam)

    origin = cam.location.copy()
    move = offset["dx"] * right + offset["dy"] * up + offset["dz"] * forward
    distance = move.length

    safe = dict(offset)

    if distance < 1e-6:
        return safe

    hit, location, obj = scene_ray_cast(origin, move, distance + margin)

    if hit:
        hit_distance = (location - origin).length
        allowed = max(0.0, hit_distance - margin)
        scale = min(1.0, allowed / distance)

        safe["dx"] = offset["dx"] * scale
        safe["dy"] = offset["dy"] * scale
        safe["dz"] = offset["dz"] * scale

        if scale < 1.0:
            hit_name = obj.name if obj is not None else "?"
            print(
                f"  collision: offset '{offset['name']}' clipped "
                f"{distance:.2f}m -> {allowed:.2f}m (hit {hit_name})"
            )

    return safe


def compute_movement_radius(focus_distance_m):
    """
    Bounded local camera motion.

    Close-up / product scene:
      focus_distance ~1m -> about 0.15m, roughly half a foot

    Room-scale scene:
      focus_distance ~10m -> about 0.5m

    Large scene:
      focus_distance ~30m -> capped near 2m
    """
    radius = 0.05 * focus_distance_m
    radius = max(radius, 0.15)
    radius = min(radius, 2.0)
    return radius


def compute_rotation_radius(focus_distance_m):
    if focus_distance_m < 2.0:
        return 3.0
    elif focus_distance_m < 8.0:
        return 5.0
    else:
        return 8.0


def make_target_view_offsets(focus_distance_m):
    r = compute_movement_radius(focus_distance_m)
    rot = compute_rotation_radius(focus_distance_m)

    return r, rot, [
        {"name": "center", "dx": 0.0, "dy": 0.0, "dz": 0.0, "yaw": 0.0, "pitch": 0.0},

        {"name": "left", "dx": -r, "dy": 0.0, "dz": 0.0, "yaw": 0.0, "pitch": 0.0},
        {"name": "right", "dx": r, "dy": 0.0, "dz": 0.0, "yaw": 0.0, "pitch": 0.0},

        {"name": "up", "dx": 0.0, "dy": 0.6 * r, "dz": 0.0, "yaw": 0.0, "pitch": 0.0},
        {"name": "down", "dx": 0.0, "dy": -0.6 * r, "dz": 0.0, "yaw": 0.0, "pitch": 0.0},

        {"name": "forward", "dx": 0.0, "dy": 0.0, "dz": 0.75 * r, "yaw": 0.0, "pitch": 0.0},
        {"name": "back", "dx": 0.0, "dy": 0.0, "dz": -0.75 * r, "yaw": 0.0, "pitch": 0.0},

        {"name": "yaw_left", "dx": 0.0, "dy": 0.0, "dz": 0.0, "yaw": -rot, "pitch": 0.0},
        {"name": "yaw_right", "dx": 0.0, "dy": 0.0, "dz": 0.0, "yaw": rot, "pitch": 0.0},

        {"name": "pitch_up", "dx": 0.0, "dy": 0.0, "dz": 0.0, "yaw": 0.0, "pitch": -0.6 * rot},
        {"name": "pitch_down", "dx": 0.0, "dy": 0.0, "dz": 0.0, "yaw": 0.0, "pitch": 0.6 * rot},

        {"name": "left_forward", "dx": -0.7 * r, "dy": 0.0, "dz": 0.5 * r, "yaw": 0.0, "pitch": 0.0},
        {"name": "right_forward", "dx": 0.7 * r, "dy": 0.0, "dz": 0.5 * r, "yaw": 0.0, "pitch": 0.0},
    ]


def apply_target_offset(cam, source_pose, offset, look_target=None):
    restore_camera_pose(cam, source_pose)

    move_camera_local(
        cam,
        dx=offset["dx"],
        dy=offset["dy"],
        dz=offset["dz"],
    )

    if look_target is not None:
        look_at(cam, look_target)

    # Apply small extra local-ish rotations after looking at target.
    if abs(offset["yaw"]) > 1e-8:
        cam.rotation_euler.rotate_axis("Z", math.radians(offset["yaw"]))

    if abs(offset["pitch"]) > 1e-8:
        cam.rotation_euler.rotate_axis("X", math.radians(offset["pitch"]))


def write_json(path, data):
    with open(bpy.path.abspath(path), "w") as f:
        json.dump(data, f, indent=2)


# ----------------------------
# ENABLE PASSES
# ----------------------------

view_layer = bpy.context.view_layer
view_layer.use_pass_z = True
view_layer.use_pass_object_index = True

object_index_map = assign_object_indices()

# ----------------------------
# MAIN LOOP
# ----------------------------

print("Camera:", camera.name, camera.location)

base_camera_pose = camera_pose_dict(camera)

mesh_distances = []
for obj in scene.objects:
    if obj.type == "MESH" and not obj.hide_render:
        d = dist_to_camera(obj)
        mesh_distances.append((obj.name, d))

mesh_distances = sorted(mesh_distances, key=lambda x: x[1])

print("Closest 20 objects:")
for name, d in mesh_distances[:20]:
    print(f"{name}: {d:.2f} m")

print("Farthest 20 objects:")
for name, d in mesh_distances[-20:]:
    print(f"{name}: {d:.2f} m")

ensure_dir(datadir)

img_i = 0

depth_bins = make_adaptive_depth_bins(num_bins=10)

for start, end in depth_bins:
    restore_camera_pose(camera, base_camera_pose)

    candidates = get_candidate_objects(start, end)

    if not candidates:
        print(f"No objects found in depth range {start:.2f}m to {end:.2f}m")
        continue

    subject = random.choice(candidates)
    subject_index = subject.pass_index
    subject_center = object_world_center(subject)

    restore_camera_pose(camera, base_camera_pose)

    curr_focus_distance = dist_to_camera(subject)

    movement_radius_m, rotation_radius_deg, target_view_offsets = make_target_view_offsets(
        curr_focus_distance
    )

    folder = os.path.join(
        datadir,
        f"img_{img_i:05d}_f{f_stop}_fl{focal_length:.1f}_fd{curr_focus_distance:.2f}"
    )

    source_dir = os.path.join(folder, "source")
    targets_dir = os.path.join(folder, "targets")

    ensure_dir(source_dir)
    ensure_dir(targets_dir)

    camera.data.sensor_width = sensor_width_mm

    # For view synthesis data, keep source/targets sharp.
    # You can add defocused renders later after geometry works.
    camera.data.dof.use_dof = False

    source_pose = camera_pose_dict(camera)

    # ----------------------------
    # 1. Source RGB
    # ----------------------------

    disable_nodes()
    render_png(os.path.join(source_dir, "rgb.png"))

    # ----------------------------
    # 2. Source Depth EXR
    # ----------------------------

    setup_depth_nodes(source_dir)
    scene.render.filepath = os.path.join(source_dir, "_compositor_dummy.png")
    bpy.ops.render.render(write_still=True)
    disable_nodes()

    # ----------------------------
    # 3. Source metadata
    # ----------------------------

    source_metadata = {
        "image_id": img_i,
        "scene_name": scene_name,
        "view_role": "source",
        "subject_name": subject.name,
        "subject_pass_index": subject_index,
        "subject_center_world": list(subject_center),
        "depth_bin_m": [start, end],
        "focus_distance_m": curr_focus_distance,
        "f_stop": f_stop,
        "focal_length_mm": focal_length,
        "sensor_width_mm": sensor_width_mm,
        "resolution": [resolution_x, resolution_y],
        "cycles_samples": samples,
        "movement_radius_m": movement_radius_m,
        "rotation_radius_deg": rotation_radius_deg,
        "movement_policy": "movement_radius_m = clamp(0.05 * focus_distance_m, 0.15, 2.0)",
        "camera": source_pose,
        "object_index_map": object_index_map,
    }

    write_json(os.path.join(source_dir, "metadata.json"), source_metadata)

    # ----------------------------
    # 4. Target nearby views
    # ----------------------------

    for view_i, requested_off in enumerate(target_view_offsets):
        off = collision_safe_offset(camera, source_pose, requested_off)

        apply_target_offset(
            camera,
            source_pose,
            off,
            look_target=subject_center,
        )

        view_name = (
            f"view_{view_i:02d}_{off['name']}"
            f"_dx{off['dx']:.2f}_dy{off['dy']:.2f}_dz{off['dz']:.2f}"
            f"_yaw{off['yaw']:.1f}_pitch{off['pitch']:.1f}"
        )

        view_dir = os.path.join(targets_dir, view_name)
        ensure_dir(view_dir)

        camera.data.dof.use_dof = False
        disable_nodes()
        render_png(os.path.join(view_dir, "rgb.png"))

        target_pose = camera_pose_dict(camera)

        target_metadata = {
            "image_id": img_i,
            "scene_name": scene_name,
            "view_role": "target",
            "view_index": view_i,
            "view_name": off["name"],
            "offset_local_m": {
                "dx": off["dx"],
                "dy": off["dy"],
                "dz": off["dz"],
            },
            "requested_offset_local_m": {
                "dx": requested_off["dx"],
                "dy": requested_off["dy"],
                "dz": requested_off["dz"],
            },
            "rotation_offset_deg": {
                "yaw": off["yaw"],
                "pitch": off["pitch"],
            },
            "f_stop": f_stop,
            "focal_length_mm": focal_length,
            "sensor_width_mm": sensor_width_mm,
            "resolution": [resolution_x, resolution_y],
            "cycles_samples": samples,
            "movement_radius_m": movement_radius_m,
            "rotation_radius_deg": rotation_radius_deg,
            "camera": target_pose,
            "source_camera": source_pose,
            "subject_name": subject.name,
            "subject_center_world": list(subject_center),
        }

        write_json(os.path.join(view_dir, "metadata.json"), target_metadata)

    restore_camera_pose(camera, source_pose)

    print(f"Saved multiview sample {folder}")

    upload_and_cleanup(folder)

    img_i += 1

restore_camera_pose(camera, base_camera_pose)

print("Done.")