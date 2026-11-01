#!/usr/bin/env python3
"""Regression check for two training-loop fixes (CPU only, synthetic data).

1. The training order changes from epoch to epoch (set_epoch acts on the loader's sampler).
2. Every model parameter is in the optimizer, including edge_weight, which module.py
   creates lazily inside forward.

Usage (from the repo root):  python scripts/check_fixes.py
Needs torch, numpy, pandas and scikit-learn. Runs run.py on a temporary copy, so the
dataset path constant in the repo is not touched.
"""
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def make_pickle(path, n_train=40, n_test=8, seed=0):
    """Synthetic CMU-MOSEI 7-class pickle in the layout dataloader.py expects."""
    rng = np.random.RandomState(seed)
    vids = ["v%03d" % i for i in range(n_train + n_test)]
    speakers, labels, text, audio, visual, sent = {}, {}, {}, {}, {}, {}
    for v in vids:
        n = rng.randint(2, 7)
        speakers[v] = ["M" if rng.rand() < 0.5 else "F" for _ in range(n)]
        labels[v] = list(rng.uniform(-3, 3, size=n))
        text[v] = rng.randn(n, 1024).astype(np.float32)
        audio[v] = rng.randn(n, 384).astype(np.float32)
        visual[v] = rng.randn(n, 35).astype(np.float32)
        sent[v] = ["x"] * n
    train_vid, test_vid = vids[:n_train], vids[n_train:]
    with open(path, "wb") as f:
        pickle.dump((vids, speakers, labels, text, audio, visual, sent,
                     train_vid, test_vid), f)


def main():
    tmp = tempfile.mkdtemp(prefix="check_fixes_")
    try:
        for name in os.listdir(REPO):
            if name.endswith(".py"):
                shutil.copy(os.path.join(REPO, name), tmp)
        pkl = os.path.join(tmp, "synthetic_mosei7.pkl")
        make_pickle(pkl)
        run_py = os.path.join(tmp, "run.py")
        with open(run_py) as f:
            src = f.read()
        src, n = re.subn(r"^CMUMOSEI7_path = .*$", "CMUMOSEI7_path = %r" % pkl,
                         src, flags=re.M)
        assert n == 1, "CMUMOSEI7_path not found in run.py"
        with open(run_py, "w") as f:
            f.write(src)

        cmd = [sys.executable, "-u", run_py, "--no_cuda", "--port", "15700",
               "--classify", "emotion", "--dataset", "CMUMOSEI7", "--epochs", "2",
               "--batch_size", "5", "--hidden_dim", "32", "--win", "5", "5",
               "--heter_n_layers", "1", "1", "1", "--shift_win", "4",
               "--modals", "tva", "--select_on", "valid", "--clean_valid"]
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=900)
        out = proc.stdout + proc.stderr
        if proc.returncode != 0:
            print(out[-3000:])
            raise SystemExit("FAIL: run.py exited with %d" % proc.returncode)

        orders = re.findall(r"train order, epoch (\d+): first 5 positions (\[[^\]]*\])", out)
        assert len(orders) == 2, "expected 2 epoch order lines, got %d" % len(orders)
        order_ok = orders[0][1] != orders[1][1]
        print("train order epoch 0:", orders[0][1])
        print("train order epoch 1:", orders[1][1])
        print("PASS: order changes between epochs" if order_ok
              else "FAIL: training order identical across epochs")

        counts = re.findall(r"optimizer params: (\d+), model params: (\d+)", out)
        assert counts, "no optimizer-count line found"
        opt_ok = all(int(a) == int(b) for a, b in counts)
        print("optimizer vs model params:", counts)
        print("PASS: every model parameter is in the optimizer" if opt_ok
              else "FAIL: optimizer is missing model parameters")

        if not (order_ok and opt_ok):
            raise SystemExit(1)
        print("ALL CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
