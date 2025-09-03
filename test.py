from comGNN import ComGNN
import argparse
import os
import time
import json
from datetime import datetime
from glob import glob

import torch
import numpy as np

from dataset import MyDataLoader, load_test_data
from train import run_batch, do_compute


def find_checkpoint(path_like: str) -> str:
    """
    Resolve a checkpoint path.
    - If path_like is a file, return it.
    - If it's a directory, choose newest *_best.pt if any, else newest *_last.pt.
    - If empty string, default to ./checkpoints using the same rule.
    """
    if not path_like:
        path_like = "checkpoints"
    if os.path.isfile(path_like):
        return path_like
    if os.path.isdir(path_like):
        bests = sorted(glob(os.path.join(path_like, "*_best.pt")), key=os.path.getmtime, reverse=True)
        lasts = sorted(glob(os.path.join(path_like, "*_last.pt")), key=os.path.getmtime, reverse=True)
        if bests:
            return bests[0]
        if lasts:
            return lasts[0]
        raise FileNotFoundError(f"No *_best.pt or *_last.pt found under {path_like!r}")
    raise FileNotFoundError(f"Checkpoint path {path_like!r} not found")


def load_model_args(ckpt_path: str) -> argparse.Namespace:
    blob = torch.load(ckpt_path, map_location="cpu")
    args_dict = blob.get("args", {}) or {}
    # Newer torch may wrap map_args as dict-like; ensure plain dict
    if hasattr(args_dict, "items") and not isinstance(args_dict, dict):
        args_dict = dict(args_dict)
    ns = argparse.Namespace(**args_dict)
    return ns


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--test_data", type=str, default="sims_harvey.npz", help="NPZ filename under ./data/")
    parser.add_argument("--time_steps", type=int, default=40, help="Window length T for sequences")
    parser.add_argument("--offset", type=int, default=10, help="Temporal offset when slicing sequences")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    # Device & checkpoint
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to .pt file or a directory with checkpoints")
    # Output
    parser.add_argument("--out_dir", type=str, default="results_test", help="Where to save metrics JSON")

    args = parser.parse_args()

    # Resolve checkpoint
    ckpt_path = find_checkpoint(args.checkpoint)
    blob = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = load_model_args(ckpt_path)

    # Model
    model = ComGNN(ckpt_args)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.to(args.device)
    model.eval()

    # Data
    test_dataset = load_test_data(args.test_data, args.offset, args.time_steps)
    test_loader = MyDataLoader(test_dataset, shuffle=False)

    # Testing
    start = time.time()
    with torch.no_grad():
        test_loss, test_mse, test_nse, test_p_r2, test_loss_nz = run_batch(
            model=model,
            optimizer=None,
            data_loader=test_loader,
            epoch_i=1,
            desc="Test",
            device=args.device,
        )
    elapsed = time.time() - start

    # Node prediction
    all_preds, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            loss, (targets, preds), mask, nz_loss = do_compute(model, batch, args.device)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            all_masks.append(mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask))

    all_preds = np.concatenate(all_preds, axis=1)
    all_targets = np.concatenate(all_targets, axis=1)
    all_masks = np.concatenate(all_masks, axis=1)

    # Prepare output
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)

    npz_path = os.path.join(args.out_dir, f"node_preds_targets_{stamp}.npz")
    np.savez_compressed(npz_path, preds=all_preds, targets=all_targets, mask=all_masks)
    print(f"Saved node-level arrays to: {npz_path}")

    meta = {
        "checkpoint": os.path.abspath(ckpt_path),
        "saved_epoch": int(blob.get("epoch", -1)),
        "best_val": float(blob.get("best_val", float('nan'))),
        "evaluated_at": stamp,
        "elapsed_sec": elapsed,
        "data": {
            "test_data": args.test_data,
            "time_steps": args.time_steps,
            "offset": args.offset,
            "batch_size": args.batch_size,
        },
        "metrics": {
            "loss": float(test_loss),
            "loss_nonzero": float(test_loss_nz),
            "mse": float(test_mse),
            "nse": float(test_nse),
            "pearson_r2": float(test_p_r2),
        },
    }
    out_path = os.path.join(args.out_dir, f"metrics_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("\n==== Test Summary ====")
    print(f"Checkpoint     : {ckpt_path}")
    print(f"Saved epoch    : {meta['saved_epoch']} (best_val={meta['best_val']:.6f})")
    print(f"Data           : {args.test_data} (T={args.time_steps}, offset={args.offset}, batch={args.batch_size})")
    print(f"Time           : {elapsed:.2f}s")
    print("Metrics        :")
    print(f"  Loss         = {meta['metrics']['loss']:.6f}")
    print(f"  Loss (nonzero)= {meta['metrics']['loss_nonzero']:.6f}")
    print(f"  MSE          = {meta['metrics']['mse']:.6f}")
    print(f"  NSE          = {meta['metrics']['nse']:.6f}")
    print(f"  Pearson R^2  = {meta['metrics']['pearson_r2']:.6f}")
    print(f"Saved metrics  : {out_path}")


if __name__ == "__main__":
    main()