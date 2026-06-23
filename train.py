import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from data_preprocessing import DEFAULT_BUCKET, DEFAULT_S3_PREFIX, FreeViewpointDataset
from models.UNet import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train the free-viewpoint Gaussian splat predictor.")
    parser.add_argument("--data-root", default=None, help="Local root containing <scene>/dataset/... samples.")
    parser.add_argument("--cache-dir", default="cache/free_viewpoint", help="Local cache for S3 samples.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene names to train on.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket for generated samples.")
    parser.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX, help="S3 prefix for generated samples.")
    parser.add_argument("--no-s3", action="store_true", help="Only use local data/cache; do not index S3.")
    parser.add_argument("--image-size", type=int, default=256, help="Square training resolution.")
    parser.add_argument("--max-depth-m", type=float, default=50.0, help="Depth normalization clamp.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for smoke tests.")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--base-channels", type=int, default=32)

    parser.add_argument("--checkpoint-dir", default="checkpoints/free_viewpoint")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--save-every", type=int, default=0, help="Also save periodic epoch checkpoints.")
    return parser.parse_args()


def make_rotation_matrices(offset):
    yaw = torch.deg2rad(offset[:, 3])
    pitch = torch.deg2rad(offset[:, 4])

    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)

    r_yaw = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )

    r_pitch = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cp, -sp], dim=-1),
            torch.stack([zeros, sp, cp], dim=-1),
        ],
        dim=-2,
    )

    return r_pitch @ r_yaw


def differentiable_splat_render(batch, outputs, background=1.0):
    """
    Render model-predicted source-pixel Gaussians into the target view.

    This is intentionally a compact training renderer: it uses the same local
    camera transform convention as utils.transform_points_camera, then bilinear
    forward-splats the predicted colors/opacity onto the target image plane.
    """
    source_rgb = batch["source_rgb"]
    depth = batch["depth_m"][:, 0]
    offset = batch["offset"]
    intrinsics = batch["intrinsics"]

    bsz, _, height, width = source_rgb.shape
    device = source_rgb.device
    dtype = source_rgb.dtype

    fx = intrinsics[:, 0].view(bsz, 1, 1)
    fy = intrinsics[:, 1].view(bsz, 1, 1)
    cx = intrinsics[:, 2].view(bsz, 1, 1)
    cy = intrinsics[:, 3].view(bsz, 1, 1)

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = xs.unsqueeze(0).expand(bsz, -1, -1)
    ys = ys.unsqueeze(0).expand(bsz, -1, -1)

    valid = torch.isfinite(depth) & (depth > 1e-6)
    z = torch.where(valid, depth, torch.ones_like(depth))
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    points = torch.stack([x, y, z], dim=-1).view(bsz, height * width, 3)
    translation = offset[:, :3].view(bsz, 1, 3)
    points = points - translation

    rotation = make_rotation_matrices(offset).to(dtype=dtype)
    points = torch.bmm(points, rotation)

    x_t = points[..., 0]
    y_t = points[..., 1]
    z_t = points[..., 2].clamp_min(1e-6)

    u = (intrinsics[:, 0:1] * (x_t / z_t) + intrinsics[:, 2:3]).view(bsz, height, width)
    v = (intrinsics[:, 1:2] * (y_t / z_t) + intrinsics[:, 3:4]).view(bsz, height, width)
    in_front = (points[..., 2].view(bsz, height, width) > 1e-6) & valid

    color = torch.clamp(source_rgb + outputs["color_residual"], 0.0, 1.0)
    radius_gain = torch.exp(outputs["radius_delta"]).clamp(0.25, 4.0)
    alpha_gain = (outputs["opacity"] * radius_gain).clamp(0.0, 0.995)

    flat_color = color.permute(0, 2, 3, 1).reshape(bsz, height * width, 3)
    flat_alpha = alpha_gain[:, 0].reshape(bsz, height * width)
    flat_valid = in_front.reshape(bsz, height * width)
    flat_u = u.reshape(bsz, height * width)
    flat_v = v.reshape(bsz, height * width)

    accum_color = torch.zeros((bsz * height * width, 3), device=device, dtype=dtype)
    accum_alpha = torch.zeros((bsz * height * width, 1), device=device, dtype=dtype)
    batch_offsets = (torch.arange(bsz, device=device) * height * width).view(bsz, 1)

    u0 = torch.floor(flat_u)
    v0 = torch.floor(flat_v)

    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ui = u0 + du
        vi = v0 + dv

        wx = 1.0 - torch.abs(flat_u - ui)
        wy = 1.0 - torch.abs(flat_v - vi)
        weight = (wx * wy).clamp_min(0.0) * flat_alpha

        mask = (
            flat_valid
            & (ui >= 0)
            & (ui < width)
            & (vi >= 0)
            & (vi < height)
            & (weight > 0)
        )

        safe_ui = ui.long().clamp(0, width - 1)
        safe_vi = vi.long().clamp(0, height - 1)
        indices = (safe_vi * width + safe_ui + batch_offsets).reshape(-1)
        weight_flat = torch.where(mask, weight, torch.zeros_like(weight)).reshape(-1, 1)
        color_flat = (flat_color * weight.unsqueeze(-1) * mask.unsqueeze(-1)).reshape(-1, 3)

        accum_color.scatter_add_(0, indices[:, None].expand(-1, 3), color_flat)
        accum_alpha.scatter_add_(0, indices[:, None], weight_flat)

    accum_color = accum_color.view(bsz, height, width, 3)
    accum_alpha = accum_alpha.view(bsz, height, width, 1).clamp(0.0, 1.0)

    image = accum_color + float(background) * (1.0 - accum_alpha)
    return image.permute(0, 3, 1, 2).clamp(0.0, 1.0)


def assert_finite_batch(batch):
    for key in ("input", "target", "source_rgb", "depth_m", "offset", "intrinsics"):
        value = batch[key]
        if torch.isnan(value).any() or torch.isinf(value).any():
            raise RuntimeError(f"Non-finite values found in batch[{key!r}]")


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    dataset = FreeViewpointDataset(
        scenes=args.scenes,
        local_dataset_dir=args.data_root,
        local_cache_dir=args.cache_dir,
        bucket_name=args.bucket,
        s3_prefix=args.s3_prefix,
        image_size=args.image_size,
        max_depth_m=args.max_depth_m,
        use_s3=not args.no_s3,
        max_samples=args.max_samples,
    )

    val_size = int(len(dataset) * args.val_split)
    if len(dataset) > 1:
        val_size = max(1, val_size)
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("Need at least two samples to create train/validation splits.")

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = UNet(in_channels=9, base_channels=args.base_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    scaler = GradScaler("cuda", enabled=(device == "cuda"))
    criterion = nn.MSELoss()

    start_epoch = 0
    best_val_loss = math.inf

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        print(f"Resumed {args.resume} at epoch {start_epoch}; best val MSE {best_val_loss:.6f}")

    checkpoint_dir = Path(args.checkpoint_dir)
    best_path = checkpoint_dir / "best_unet_splat.pth"
    debug_dir = Path("debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            assert_finite_batch(batch)
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(device == "cuda")):
                outputs = model(batch["input"])
                prediction = differentiable_splat_render(batch, outputs)
                loss = criterion(prediction, batch["target"])
                loss = loss + 0.02 * F.l1_loss(outputs["color_residual"], torch.zeros_like(outputs["color_residual"]))
                loss += 0.001 * outputs["opacity"].mean()
                loss += 0.001 * torch.mean(outputs["radius_delta"] ** 2)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, batch {batch_idx}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                assert_finite_batch(batch)
                batch = {
                    key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                outputs = model(batch["input"])
                prediction = differentiable_splat_render(batch, outputs)
                val_loss += criterion(prediction, batch["target"]).item()

                if batch_idx == 0:
                    save_image(
                        torch.cat(
                            [
                                batch["source_rgb"][0],
                                prediction[0],
                                batch["target"][0],
                            ],
                            dim=2,
                        ),
                        debug_dir / f"epoch_{epoch + 1:04d}.png",
                    )

        avg_val_loss = val_loss / max(1, len(val_loader))
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train MSE: {avg_train_loss:.6f} | "
            f"Val MSE: {avg_val_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(best_path, model, optimizer, scheduler, epoch + 1, best_val_loss, args)
            print(f"Saved new best checkpoint: {best_path}")

        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pth",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_val_loss,
                args,
            )

    print("Training complete!")


if __name__ == "__main__":
    main()
