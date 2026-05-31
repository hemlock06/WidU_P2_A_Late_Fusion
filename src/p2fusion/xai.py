"""late fusion 게이트 XAI — gated fusion 판정 근거 추출·설명.

GatedFusionModel은 forward마다 다음을 내보낸다(P3 활용 목적 설계):
  gate_weights[B,3]      동적 모달 가중치 (ecg·imu·spo2) — 게이트넷 산출(≠reliability)
  conf_per_modality[B,3] 각 expert 확신도 (단독 softmax max)
  unimodal_logits[B,3,5] 각 모달 독립 5분류 예측

→ "어느 모달이 이 판정을 주도했나(게이트) · 각 모달은 무엇을 얼마나 확신했나(conf·단독예측)"를 요약·설명.

cross-modal attention XAI와 대비:
  attention = 교차모달 [3,3](모달 간 상호작용 정밀). late fusion = 모달별 게이트·단독예측(거친·명확).
  late fusion은 교차모달 상호작용은 못 보지만, 어느 expert가 주도했는지는 직접 드러낸다.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

MOD = ["ECG", "IMU", "SpO2"]
_CLASS_KO = ["정상(안정)", "정상(활동)", "심혈관 응급", "낙상·충격", "저산소"]
_CLASS_PRIMARY_MOD = [None, "IMU", "ECG", "IMU", "SpO2"]
# ecg_aux: 0-4 cardiac_probs, 5 emergency_score, 6 reliability, 7 gate_tier, 8 hr, 9 rhythm
_AUX_REL = 6


def _to_batch(arrays: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in arrays.items()}


@torch.no_grad()
def collect_gate(model, arrays: Dict[str, np.ndarray], device, batch_size: int = 1024):
    """arrays(ecg_emb·ecg_aux·imu·spo2·mask) → (gate_w[N,3], conf[N,3], uni_logits[N,3,5], pred[N]).

    GatedFusionModel 전용 (gate_weights·conf_per_modality·unimodal_logits 출력 필요).
    """
    model.eval()
    n = len(arrays["ecg_emb"])
    GW, CF, UL, PR = [], [], [], []
    for i in range(0, n, batch_size):
        sub = {k: v[i:i + batch_size] for k, v in arrays.items()}
        out = model(_to_batch(sub, device))
        for key in ("gate_weights", "conf_per_modality", "unimodal_logits"):
            if key not in out:
                raise ValueError(f"모델이 {key}를 출력하지 않음 (gated fusion 전용 XAI)")
        GW.append(out["gate_weights"].cpu().numpy())
        CF.append(out["conf_per_modality"].cpu().numpy())
        UL.append(out["unimodal_logits"].cpu().numpy())
        PR.append(out["logits"].argmax(-1).cpu().numpy())
    return (np.concatenate(GW), np.concatenate(CF),
            np.concatenate(UL), np.concatenate(PR))


def summarize_gate(gate_w: np.ndarray) -> np.ndarray:
    """gate_w[N,3] → 평균 모달 기여[3] (정규화)."""
    m = gate_w.mean(axis=0)
    return m / (m.sum() + 1e-8)


def format_gate(gate_w: np.ndarray, conf: np.ndarray, title: str = "") -> str:
    mg, mc = gate_w.mean(0), conf.mean(0)
    lines = []
    if title:
        lines.append(title)
    lines.append("  모달 기여(게이트 평균):  " + "  ".join(f"{m}={mg[i]:.3f}" for i, m in enumerate(MOD)))
    lines.append("  모달 확신도(평균):       " + "  ".join(f"{m}={mc[i]:.3f}" for i, m in enumerate(MOD)))
    return "\n".join(lines)


def generate_gate_explanation(pred_class: int,
                              gate_w: np.ndarray,
                              conf: np.ndarray,
                              unimodal_logits: np.ndarray,
                              ecg_aux: np.ndarray) -> str:
    """단일 판정 자연어 설명 — gate_weights[3]·conf[3]·unimodal_logits[3,5]·ecg_aux[10].

    pred_class: 0정상안정 1정상활동 2심혈관 3낙상 4저산소
    """
    dom = int(np.argmax(gate_w))
    rel = float(ecg_aux[_AUX_REL])
    uni_votes = [_CLASS_KO[int(np.argmax(unimodal_logits[i]))] for i in range(3)]

    lines = [f"[판정] {_CLASS_KO[pred_class]}"]
    lines.append("[모달 기여(게이트)] " + "  ".join(f"{m} {gate_w[i]:.0%}" for i, m in enumerate(MOD)))
    lines.append("[모달 확신도]       " + "  ".join(f"{m} {conf[i]:.0%}" for i, m in enumerate(MOD)))
    lines.append(f"  → {MOD[dom]} 주도 (게이트 {gate_w[dom]:.0%}, 확신 {conf[dom]:.0%}, "
                 f"단독예측={uni_votes[dom]})")

    # 기대 모달과 비교
    expected = _CLASS_PRIMARY_MOD[pred_class]
    if expected and MOD[dom] != expected:
        lines.append(f"     (기대 1차모달 {expected}와 불일치 — 결측/신호불량으로 게이트가 대체 모달 선택)")

    # 신호불량 주의
    if rel > 0.6 and dom == 0:
        lines.append(f"[주의] ECG 주도이나 reliability {rel:.2f} 높음(신호 불량) — 판정 신뢰도 낮음, 재확인 권고.")

    return "\n".join(lines)


def gate_report(model, arrays_by_group: Dict[str, Dict[str, np.ndarray]], device) -> str:
    """그룹별(클래스/confounder) 게이트 기여 요약 — 어느 모달이 각 그룹을 주도하나."""
    blocks = []
    for name, arrays in arrays_by_group.items():
        gw, cf, ul, pr = collect_gate(model, arrays, device)
        dist = np.bincount(pr, minlength=5).tolist()
        title = f"[{name}] n={len(pr)}  pred 분포={dist}"
        blocks.append(format_gate(gw, cf, title))
    return "\n\n".join(blocks)
