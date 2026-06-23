import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import find_depth_file, get_intrinsics, load_depth, load_rgb


DEFAULT_BUCKET = "tejas-blender-bucket"
DEFAULT_S3_PREFIX = "defocus-dataset-multiview"


def _maybe_import_boto3():
    try:
        import boto3
    except ImportError:
        return None
    return boto3


def _resize_rgb(image, size):
    if image.shape[:2] == (size, size):
        return image.astype(np.float32)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def _resize_depth(depth, size):
    if depth.shape[:2] == (size, size):
        return depth.astype(np.float32)
    return cv2.resize(depth, (size, size), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def _as_chw(image):
    return np.transpose(image, (2, 0, 1)).astype(np.float32)


class FreeViewpointDataset(Dataset):
    """
    Dataset for the multiview samples produced by scene_data.py.

    Expected sample layout:
        <root>/<scene>/dataset/<sample>/source/rgb.png
        <root>/<scene>/dataset/<sample>/source/depth_*.exr
        <root>/<scene>/dataset/<sample>/source/metadata.json
        <root>/<scene>/dataset/<sample>/targets/<view>/rgb.png
        <root>/<scene>/dataset/<sample>/targets/<view>/metadata.json

    Each item returns a dict with:
        input:      9xHxW tensor, RGB + normalized depth + target offset maps
        target:     3xHxW tensor, target-view RGB
        source_rgb: 3xHxW tensor, source RGB for differentiable splatting
        depth_m:    1xHxW tensor, metric source depth
        offset:     [dx, dy, dz, yaw, pitch] tensor
        intrinsics: [fx, fy, cx, cy] tensor at the resized training resolution
    """

    def __init__(
        self,
        scenes=None,
        local_dataset_dir=None,
        local_cache_dir="cache/free_viewpoint",
        bucket_name=DEFAULT_BUCKET,
        s3_prefix=DEFAULT_S3_PREFIX,
        image_size=256,
        max_depth_m=50.0,
        use_s3=True,
        max_samples=None,
    ):
        self.scenes = tuple(scenes) if scenes else None
        self.local_dataset_dir = Path(local_dataset_dir) if local_dataset_dir else None
        self.local_cache_dir = Path(local_cache_dir)
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix.strip("/")
        self.image_size = int(image_size)
        self.max_depth_m = float(max_depth_m)
        self.use_s3 = use_s3

        self.s3 = None
        self.samples = []

        if self.local_dataset_dir:
            self.samples.extend(self._index_local_dataset(self.local_dataset_dir))

        if not self.samples:
            cache_root = self.local_cache_dir / self.s3_prefix
            if cache_root.exists():
                self.samples.extend(self._index_local_dataset(cache_root))

        if not self.samples and self.use_s3:
            self.samples.extend(self._index_s3_dataset())

        if max_samples is not None:
            self.samples = self.samples[: int(max_samples)]

        if not self.samples:
            roots = []
            if self.local_dataset_dir:
                roots.append(str(self.local_dataset_dir))
            roots.append(str(self.local_cache_dir / self.s3_prefix))
            raise RuntimeError(
                "No multiview source/target pairs found. Checked local roots "
                f"{roots}; S3 lookup was {'enabled' if self.use_s3 else 'disabled'}."
            )

        print(f"Found {len(self.samples)} free-viewpoint training pairs.")

    def __len__(self):
        return len(self.samples)

    def _scene_allowed(self, scene):
        return self.scenes is None or scene in self.scenes

    def _index_local_dataset(self, root):
        samples = []
        root = Path(root)

        scene_roots = []
        if (root / "dataset").is_dir():
            scene_roots.append(root)
        scene_roots.extend(p for p in sorted(root.iterdir()) if (p / "dataset").is_dir())

        for scene_root in scene_roots:
            scene = scene_root.name
            if not self._scene_allowed(scene):
                continue

            source_dirs = sorted((scene_root / "dataset").glob("*/source"))
            for source_dir in source_dirs:
                samples.extend(self._index_local_source(scene, source_dir))

        return samples

    def _index_local_source(self, scene, source_dir):
        samples = []
        source_dir = Path(source_dir)
        sample_dir = source_dir.parent
        targets_dir = sample_dir / "targets"
        if not targets_dir.is_dir():
            return samples

        for target_dir in sorted(p for p in targets_dir.iterdir() if p.is_dir()):
            if not (target_dir / "rgb.png").is_file():
                continue
            if not (target_dir / "metadata.json").is_file():
                continue
            if not (source_dir / "rgb.png").is_file():
                continue
            if not (source_dir / "metadata.json").is_file():
                continue

            samples.append(
                {
                    "kind": "local",
                    "scene": scene,
                    "sample_name": sample_dir.name,
                    "target_name": target_dir.name,
                    "source_dir": source_dir,
                    "target_dir": target_dir,
                }
            )

        return samples

    def _s3_client(self):
        if self.s3 is None:
            boto3 = _maybe_import_boto3()
            if boto3 is None:
                raise RuntimeError("boto3 is required for S3 dataset indexing.")
            self.s3 = boto3.client("s3")
        return self.s3

    def _list_common_prefixes(self, prefix):
        paginator = self._s3_client().get_paginator("list_objects_v2")
        out = []
        for page in paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=prefix,
            Delimiter="/",
        ):
            out.extend(obj["Prefix"] for obj in page.get("CommonPrefixes", []))
        return out

    def _discover_s3_scenes(self):
        root_prefix = f"{self.s3_prefix}/"
        scene_prefixes = self._list_common_prefixes(root_prefix)
        scenes = [Path(prefix.rstrip("/")).name for prefix in scene_prefixes]
        return sorted(scene for scene in scenes if self._scene_allowed(scene))

    def _index_s3_dataset(self):
        samples = []
        scenes = self.scenes or self._discover_s3_scenes()

        for scene in scenes:
            dataset_prefix = f"{self.s3_prefix}/{scene}/dataset/"
            for sample_prefix in self._list_common_prefixes(dataset_prefix):
                sample_name = Path(sample_prefix.rstrip("/")).name
                targets_prefix = f"{sample_prefix}targets/"

                for target_prefix in self._list_common_prefixes(targets_prefix):
                    target_name = Path(target_prefix.rstrip("/")).name
                    samples.append(
                        {
                            "kind": "s3",
                            "scene": scene,
                            "sample_name": sample_name,
                            "target_name": target_name,
                            "sample_prefix": sample_prefix,
                            "target_prefix": target_prefix,
                        }
                    )

        return samples

    def _download_if_missing(self, key, local_path):
        local_path = Path(local_path)
        if local_path.exists():
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._s3_client().download_file(self.bucket_name, key, str(local_path))
        return local_path

    def _find_s3_depth_key(self, source_prefix):
        response = self._s3_client().list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=source_prefix,
        )
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".exr", ".npy")):
                return key
        raise FileNotFoundError(f"No depth file found under s3://{self.bucket_name}/{source_prefix}")

    def _materialize_s3_sample(self, sample):
        cache_sample = (
            self.local_cache_dir
            / self.s3_prefix
            / sample["scene"]
            / "dataset"
            / sample["sample_name"]
        )
        source_dir = cache_sample / "source"
        target_dir = cache_sample / "targets" / sample["target_name"]

        source_prefix = f"{sample['sample_prefix']}source/"
        target_prefix = sample["target_prefix"]

        self._download_if_missing(f"{source_prefix}rgb.png", source_dir / "rgb.png")
        self._download_if_missing(f"{source_prefix}metadata.json", source_dir / "metadata.json")

        depth_key = self._find_s3_depth_key(source_prefix)
        self._download_if_missing(depth_key, source_dir / Path(depth_key).name)

        self._download_if_missing(f"{target_prefix}rgb.png", target_dir / "rgb.png")
        self._download_if_missing(f"{target_prefix}metadata.json", target_dir / "metadata.json")

        return source_dir, target_dir

    def _sample_dirs(self, sample):
        if sample["kind"] == "local":
            return sample["source_dir"], sample["target_dir"]
        return self._materialize_s3_sample(sample)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        source_dir, target_dir = self._sample_dirs(sample)

        source_rgb = _resize_rgb(load_rgb(str(source_dir / "rgb.png")), self.image_size)
        target_rgb = _resize_rgb(load_rgb(str(target_dir / "rgb.png")), self.image_size)
        depth_m = _resize_depth(load_depth(find_depth_file(str(source_dir))), self.image_size)

        with open(source_dir / "metadata.json", "r") as f:
            source_metadata = json.load(f)
        with open(target_dir / "metadata.json", "r") as f:
            target_metadata = json.load(f)

        height, width = depth_m.shape
        fx, fy, cx, cy = get_intrinsics(source_metadata, width=width, height=height)

        offset_local = target_metadata["offset_local_m"]
        rotation_offset = target_metadata["rotation_offset_deg"]
        offset = np.array(
            [
                offset_local["dx"],
                offset_local["dy"],
                offset_local["dz"],
                rotation_offset["yaw"],
                rotation_offset["pitch"],
            ],
            dtype=np.float32,
        )

        depth_norm = np.clip(depth_m, 0.0, self.max_depth_m) / self.max_depth_m
        condition_maps = np.broadcast_to(
            offset[:, None, None],
            (5, height, width),
        ).astype(np.float32)

        x = np.concatenate(
            [
                _as_chw(source_rgb),
                depth_norm[None, :, :].astype(np.float32),
                condition_maps,
            ],
            axis=0,
        )

        item = {
            "input": torch.from_numpy(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)),
            "target": torch.from_numpy(_as_chw(target_rgb)),
            "source_rgb": torch.from_numpy(_as_chw(source_rgb)),
            "depth_m": torch.from_numpy(depth_m[None, :, :].astype(np.float32)),
            "offset": torch.from_numpy(offset),
            "intrinsics": torch.tensor([fx, fy, cx, cy], dtype=torch.float32),
            "sample_id": f"{sample['scene']}/{sample['sample_name']}/{sample['target_name']}",
        }

        return item


# Backwards-compatible alias for older notebooks/scripts while the project
# transitions to the free-viewpoint naming.
MultiviewDataset = FreeViewpointDataset
DefocusDataset = FreeViewpointDataset
