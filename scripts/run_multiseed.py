"""reliability 통합 방식 다중시드 확정 실험.

run_ablation.py의 핵심 변형 중 reliability 처리 방식 3종(E1/E2/E3)을 여러 시드로
반복 학습해 cardiac recall·결측 강건성 델타가 노이즈가 아닌지 확정한다.

사용:
    python scripts/run_multiseed.py --epochs 40 --seeds 42,1,7
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2fusion.data.dataset import P2Dataset
from p2fusion.models.concat_mlp import ConcatMLP
from p2fusion.models.gated_fusion import GatedFusionModel
from p2fusion.schema import CLASS_NAMES, NUM_CLASSES

# run_ablation의 학습/평가 유틸 재사용
from run_ablation import (DATA_DIR, DEVICE, macro_f1, per_class_recall,
                          predict, train_one)


def build(key, dropout):
    if key == "E0_concat":
        return ConcatMLP(dropout_p=dropout)
    if key == "E5_conf_routed":
        return GatedFusionModel(reliability_mode="feature", gate_input_norm=True,
                                dropout=dropout, aux_loss_weight=0.3,
                                gate_mode="conf_routed", temperature=0.15)
    cfg = {
        "E1_gate_norel":    ("none",      True),
        "E2_gate_relfeat":  ("feature",   True),
        "E3_gate_hardmult": ("hard_mult", True),
    }[key]
    return GatedFusionModel(reliability_mode=cfg[0], gate_input_norm=cfg[1],
                            dropout=dropout, aux_loss_weight=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--variants", default="E0_concat,E1_gate_norel,E2_gate_relfeat,E3_gate_hardmult")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--dropout-mod", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    print(f"Device: {DEVICE} | epochs={args.epochs} | seeds={seeds}")

    val_ds  = P2Dataset(DATA_DIR / "p2_synth_v1_val.npz")
    test_ds = P2Dataset(DATA_DIR / "p2_synth_v1_test.npz")
    pin = torch.cuda.is_available()
    val_loader  = DataLoader(val_ds,  batch_size=512, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=512, pin_memory=pin)

    # metric_key -> {variant -> [values per seed]}
    metrics = {m: {v: [] for v in variants}
               for m in ["clean", "cardiac", "drop_ecg", "drop_imu", "drop_spo2"]}

    for seed in seeds:
        print(f"\n{'#'*70}\n# SEED {seed}\n{'#'*70}")
        train_ds = P2Dataset(DATA_DIR / "p2_synth_v1_train.npz",
                             modality_dropout_p=args.dropout_mod, seed=seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=pin)
        for v in variants:
            torch.manual_seed(seed); np.random.seed(seed)
            model = build(v, args.dropout)
            t0 = time.time()
            model, _ = train_one(model, train_loader, val_loader, args.epochs, args.lr)

            preds, labels = predict(model, test_loader)
            clean, _ = macro_f1(preds, labels)
            recall = per_class_recall(preds, labels)
            miss = {}
            for m_idx, m_name in [(0, "drop_ecg"), (1, "drop_imu"), (2, "drop_spo2")]:
                mp, ml = predict(model, test_loader, drop_modality=m_idx)
                miss[m_name], _ = macro_f1(mp, ml)

            metrics["clean"][v].append(clean)
            metrics["cardiac"][v].append(recall[2])
            for mn in ["drop_ecg", "drop_imu", "drop_spo2"]:
                metrics[mn][v].append(miss[mn])
            print(f"  [{v:<18}] clean={clean:.4f} cardiac={recall[2]:.3f} "
                  f"-IMU={miss['drop_imu']:.3f} -SpO2={miss['drop_spo2']:.3f} "
                  f"({time.time()-t0:.0f}s)")

    # ── 집계 ──
    def ms(vals):
        a = np.array(vals)
        return a.mean(), a.std()

    print(f"\n\n{'='*100}")
    print(f"  다중시드 요약 (n={len(seeds)} seeds, mean±std)")
    print(f"{'='*100}")
    cols = ["clean", "cardiac", "drop_ecg", "drop_imu", "drop_spo2"]
    hdr = f"{'variant':<20}" + "".join(f"{c:>16}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for v in variants:
        row = f"{v:<20}"
        for c in cols:
            mean, std = ms(metrics[c][v])
            row += f"{mean:>10.4f}±{std:.3f}"
        print(row)
    print("-" * len(hdr))

    # 핵심 대비: E2 vs E3 (cardiac, drop_imu)
    if "E2_gate_relfeat" in variants and "E3_gate_hardmult" in variants:
        for c in ["cardiac", "drop_imu", "clean"]:
            e2m, _ = ms(metrics[c]["E2_gate_relfeat"])
            e3m, _ = ms(metrics[c]["E3_gate_hardmult"])
            print(f"  Δ({c}) E2−E3 = {e2m-e3m:+.4f}")


if __name__ == "__main__":
    main()
