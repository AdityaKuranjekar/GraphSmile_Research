#!/usr/bin/env python3
"""Gap plan, Phase 1.1 and 1.2: CMU-MOSEI class counts and feature audit.

CPU only, no training, no GPU. Reads the CMU-MOSEI pickle and writes a JSON report.

Usage (from the repo root):
    python scripts/gap_phase1_mosei_audit.py /path/to/CMUMOSEI7.pkl [out.json]

The pickle path is CMUMOSEI7_path in run.py.
"""
import json
import os
import pickle
import sys

import numpy as np

CLASSES = ["HNeg", "Neg", "WNeg", "Neutral", "WPos", "Pos", "HPos"]
PAPER_T2 = {"train_val_dialogues": 2549, "train_val_utterances": 18198,
            "test_dialogues": 646, "test_utterances": 4662}


def emo7(a):  # same bins as dataloader.CMUMOSEIDataset7
    if a < -2: return 0
    if a < -1: return 1
    if a < 0: return 2
    if a == 0: return 3
    if a <= 1: return 4
    if a <= 2: return 5
    return 6


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results/experiments/gap-phase1-mosei-audit.json"
    with open(path, "rb") as f:
        (vids, speakers, labels, text, audio, visual, _sent,
         train_vid, test_vid) = pickle.load(f, encoding="latin1")

    rep = {"file": path, "dialogues_in_pickle": len(vids),
           "train_test_overlap": len(set(train_vid) & set(test_vid))}

    for split, keys in (("train_val", train_vid), ("test", test_vid)):
        lens = [len(labels[v]) for v in keys]
        cls = np.zeros(7, dtype=int)
        sen = np.zeros(3, dtype=int)  # neg, zero, pos
        for v in keys:
            for a in labels[v]:
                a = float(a)
                cls[emo7(a)] += 1
                sen[0 if a < 0 else (1 if a == 0 else 2)] += 1
        rep[split] = {
            "dialogues": len(keys),
            "utterances": int(sum(lens)),
            "dialogue_len_mean": round(float(np.mean(lens)), 2),
            "dialogue_len_median": float(np.median(lens)),
            "dialogue_len_max": int(max(lens)),
            "share_dialogues_len1": round(float(np.mean(np.array(lens) == 1)), 3),
            "emo7_counts": dict(zip(CLASSES, cls.tolist())),
            "emo7_share": dict(zip(CLASSES, np.round(cls / cls.sum(), 4).tolist())),
            "sentiment3_counts": {"neg": int(sen[0]), "zero": int(sen[1]), "pos": int(sen[2])},
        }
    rep["paper_table2_reference"] = PAPER_T2

    mods = {"text": text, "audio": audio, "visual": visual}
    rep["modalities"] = {}
    for m, feats in mods.items():
        shapes, nan_r, inf_r, zero_r, rows, miss, mism = set(), 0, 0, 0, 0, 0, 0
        for v in vids:
            if v not in feats:
                miss += 1
                continue
            a = np.asarray(feats[v], dtype=np.float64)
            a2 = a.reshape(a.shape[0], -1)
            rows += a2.shape[0]
            shapes.add(tuple(a.shape[1:]))
            if a2.shape[0] != len(labels[v]):
                mism += 1
            nan_r += int(np.isnan(a2).any(axis=1).sum())
            inf_r += int(np.isinf(a2).any(axis=1).sum())
            zero_r += int((a2 == 0).all(axis=1).sum())
        rep["modalities"][m] = {
            "missing_dialogues": miss,
            "feature_shapes": sorted(str(list(s)) for s in shapes),
            "rows": rows,
            "rows_with_nan": nan_r,
            "rows_with_inf": inf_r,
            "rows_all_zero": zero_r,
            "dialogues_length_mismatch_vs_labels": mism,
        }

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    print("written:", out)


if __name__ == "__main__":
    main()
