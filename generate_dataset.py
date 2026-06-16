"""
Orchestrates multiview dataset generation across every Blender scene.

A single scene-agnostic script, scene_data.py, is run once per scene inside
Blender. scene_data.py derives the scene name from the folder containing the
.blend file, renders into "//dataset/" (which resolves to <scene_folder>/dataset/),
and uploads each rendered sample to
s3://<S3_BUCKET>/<S3_PREFIX>/<scene>/dataset/<sample>/ as it is produced,
deleting it locally afterwards.

Scenes live in S3, one prefix per scene:
    s3://<S3_BUCKET>/<S3_PREFIX>/<scene>/<scene>.blend
    s3://<S3_BUCKET>/<S3_PREFIX>/<scene>/dataset/...

Designed to run on a VM where the .blend files are NOT stored locally. For each
scene this script:
    1. Downloads the scene's .blend file from
       s3://<S3_BUCKET>/<S3_PREFIX>/<scene>/ into scenes/<scene>/.
    2. Runs scene_data.py inside Blender (headless) against that .blend.
    3. Deletes the downloaded .blend (and any leftover local dataset/) to free
       disk before the next scene.

Usage:
    python generate_dataset.py                 # run every scene found in S3
    python generate_dataset.py cafe house      # run only the named scenes
    python generate_dataset.py --keep-local    # don't delete the downloaded blend / leftovers
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

# ----------------------------
# CONFIG
# ----------------------------

# Local scratch directory where per-scene folders / .blend files are downloaded.
SCENES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes")

# The single scene-agnostic Blender script that renders any scene.
SCENE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_data.py")

# Where scene_data.py writes its renders ("//dataset/" in the blend file
# resolves to <scene_folder>/dataset/).
DATASET_SUBDIR = "dataset"

# S3 location: s3://<bucket>/<prefix>/<scene>/...
# Must match S3_BUCKET / S3_PREFIX in scene_data.py.
S3_BUCKET = "tejas-blender-bucket"
S3_PREFIX = "defocus-dataset-multiview"

# Scenes to skip during discovery (e.g. broken or unwanted scenes).
IGNORED_SCENES = {"bottle"}


def resolve_blender():
    """Locate the Blender executable.

    Resolution order:
      1. The BLENDER env var (explicit override).
      2. blender on PATH.
      3. macOS app bundle.
      4. A Blender install under /workspace (the VM layout, e.g.
         /workspace/blender-4.2.0-linux-x64/blender).
    """
    env = os.environ.get("BLENDER")
    if env:
        return env

    on_path = shutil.which("blender")
    if on_path:
        return on_path

    macos = "/Applications/Blender.app/Contents/MacOS/Blender"
    if os.path.exists(macos):
        return macos

    workspace_candidates = (
        glob.glob("/workspace/blender")
        + glob.glob("/workspace/blender*/blender")
        + glob.glob("/workspace/**/blender", recursive=True)
    )
    for cand in workspace_candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand

    # Fall back to the bare name; will error clearly at run time if missing.
    return "blender"


# Blender executable. Override with the BLENDER env var if needed.
BLENDER = resolve_blender()


# ----------------------------
# HELPERS
# ----------------------------

def discover_scenes():
    """Return a sorted list of scene names discovered under the S3 prefix.

    Each scene is a "directory" (common prefix) directly under
    s3://<S3_BUCKET>/<S3_PREFIX>/.
    """
    s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/"

    result = subprocess.run(
        ["aws", "s3", "ls", s3_uri],
        check=True,
        capture_output=True,
        text=True,
    )

    scenes = []
    for line in result.stdout.splitlines():
        # Directory lines look like: "                           PRE cafe/"
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "PRE":
            name = parts[1].rstrip("/")
            if name and name not in IGNORED_SCENES:
                scenes.append(name)

    return sorted(scenes)


def find_blend_file(scene_dir):
    """Find a local .blend file in a scene folder (ignoring Blender's .blend1 backups)."""
    candidates = [
        f for f in glob.glob(os.path.join(scene_dir, "*.blend"))
        if not f.endswith(".blend1")
    ]
    return candidates[0] if candidates else None


def s3_find_blend_name(scene):
    """List the scene's S3 prefix and return the .blend file name (ignoring .blend1)."""
    s3_prefix = f"s3://{S3_BUCKET}/{S3_PREFIX}/{scene}/"

    result = subprocess.run(
        ["aws", "s3", "ls", s3_prefix],
        check=True,
        capture_output=True,
        text=True,
    )

    for line in result.stdout.splitlines():
        # File lines look like: "2026-05-30 14:59:00   37117339 cafe.blend"
        # Directory lines look like: "                           PRE dataset/"
        name = line.split()[-1]
        if name.endswith(".blend"):
            return name

    return None


def download_blend(scene):
    """Download the scene's .blend file from S3 into scenes/<scene>/ and return its local path.

    The .blend is placed inside a folder named exactly <scene> so that
    scene_data.py (which derives the scene name from the .blend's parent folder)
    resolves the correct scene name.
    """
    scene_dir = os.path.join(SCENES_DIR, scene)

    # Use a local copy if one already exists (e.g. running on the dev machine).
    local = find_blend_file(scene_dir)
    if local:
        print(f"[{scene}] Using local blend file: {local}")
        return local

    blend_name = s3_find_blend_name(scene)
    if not blend_name:
        raise FileNotFoundError(
            f"No .blend file found at s3://{S3_BUCKET}/{S3_PREFIX}/{scene}/"
        )

    os.makedirs(scene_dir, exist_ok=True)
    local_path = os.path.join(scene_dir, blend_name)
    s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/{scene}/{blend_name}"

    print(f"[{scene}] Downloading {s3_uri} -> {local_path}")
    subprocess.run(["aws", "s3", "cp", s3_uri, local_path], check=True)

    return local_path


def run_scene(scene, blend_file):
    """Run the shared scene_data.py script inside Blender (headless)."""
    cmd = [
        BLENDER,
        "--background",
        blend_file,
        "--python",
        SCENE_SCRIPT,
    ]

    print(f"\n[{scene}] Rendering with Blender")
    print(f"[{scene}] $ {' '.join(cmd)}")

    subprocess.run(cmd, check=True)


def delete_local_files(scene, blend_file=None):
    """Delete the local dataset/ folder and the downloaded .blend for a scene."""
    local_dir = os.path.join(SCENES_DIR, scene, DATASET_SUBDIR)

    if os.path.isdir(local_dir):
        print(f"[{scene}] Deleting local dataset: {local_dir}")
        shutil.rmtree(local_dir)

    if blend_file and os.path.isfile(blend_file):
        print(f"[{scene}] Deleting downloaded blend: {blend_file}")
        os.remove(blend_file)


# ----------------------------
# MAIN
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate the multiview defocus dataset across all Blender scenes.")
    parser.add_argument("scenes", nargs="*", help="Specific scene names to run (default: all scenes found in S3).")
    parser.add_argument("--keep-local", action="store_true", help="Do not delete the downloaded blend / leftover output.")
    args = parser.parse_args()

    if args.scenes:
        scenes = args.scenes
    else:
        scenes = discover_scenes()

    if not scenes:
        print(f"No scenes found at s3://{S3_BUCKET}/{S3_PREFIX}/")
        sys.exit(1)

    if not os.path.isfile(SCENE_SCRIPT):
        print(f"Scene script not found: {SCENE_SCRIPT}")
        sys.exit(1)

    print(f"Blender: {BLENDER}")
    print(f"Scene script: {SCENE_SCRIPT}")
    print(f"Scenes to process ({len(scenes)}): {', '.join(scenes)}")

    failures = []

    for scene in scenes:
        try:
            blend_file = download_blend(scene)

            run_scene(scene, blend_file)

            if not args.keep_local:
                delete_local_files(scene, blend_file)

            print(f"[{scene}] Done.")

        except subprocess.CalledProcessError as e:
            print(f"[{scene}] FAILED (exit code {e.returncode}). Keeping local files.")
            failures.append(scene)
        except Exception as e:
            print(f"[{scene}] FAILED: {e}. Keeping local files.")
            failures.append(scene)

    print("\n========================================")
    if failures:
        print(f"Completed with failures in: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All scenes generated successfully!")


if __name__ == "__main__":
    main()
