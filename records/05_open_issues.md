# 이슈 트래커 (Open Issues)

---

## 열린 이슈

### I-5. ECG 임베딩 과적합 — val/test 격차 0.107 (진행 중)

vf 재학습 결과 val F1=0.962, test F1=0.855. 격차 원인: NSR 768차원 임베딩이
P1-test 130개 풀에서 일반화 실패. 비심장 클래스(rest/active/impact/hypoxia)가 타격.
가설: raw 임베딩 제거(ecg_aux 10차원만)로 과적합 해소 → `--no-embedding` 실험 진행 중.
- 닫히면: 임베딩 제거 확정 → 재학습 → 정직한 test 성능
- 안 닫히면: NSR pool 크기 불균형이 원인 → CPSC split 재조정 검토

### I-1. sim-to-real 갭 — 조건부 독립 합성의 과분리 경향

방법 A(조건부 독립 조립)는 모달리티 간 상관·공유 nuisance가 없어 **결합 시 과도하게 분리**되는
경향(초기 선형 macro-F1=1.0). 측정노이즈+hard case로 천장을 0.94로 낮췄으나, 이는 인위적 보정.
- 완화책: (2단계) 실데이터 통계로 사전분포 정합 + 동시수집 데이터(PTT-PPG)에서 모달리티 상관 참조.
- 보강책: 방법 B(시뮬레이터)로 모달리티 간 동역학(낙상 충격↔놀람 빈맥 타이밍 등) 주입.
- 평가 시 합성-실 갭을 반드시 명시.

### ✅ I-3. Harespod SpO2 절대 스케일 — 재앵커링으로 수렴 검증

고도-SpO2 앵커(2.0km→95%, 4.0km→85%)로 역산 성공.
역산 절대값이 문헌 prior(nadir 84%, mean 89%, std 2.8)와 ~1-3% 내 수렴.
한계: 앵커가 문헌값이라 순환적 — 독립 측정 아닌 "수렴 검증"으로 정의.
임상 응급(<80%) 범위 Harespod 미포함 → 극단 저산소 검증 불가.

### ✅ I-3 (구버전). Harespod SpO2 절대 스케일 소실 (열림, 우회됨)

Harespod 공개 SpO2는 피험자별 min-max 정규화 + dropout 아티팩트 → 절대% 복원 불가.
- 우회: SpO2 prior는 임상 문헌값 유지(`01_design_decisions.md §5`).
- 향후: 절대% SpO2 동시수집 데이터 확보 시 재보정 (현재 미발견).

### I-4. SpO2 채널 동시수집 데이터 여전히 부재

PTT-PPG는 IMU+ECG는 동시수집이나 SpO2 수치 채널 없음(pleth=PPG 파형만).
→ ECG+IMU 모달리티 상관은 PTT-PPG로 검증 가능하나, SpO2 결합 상관은 여전히 합성 의존.

---

## 완료된 이슈

### ✅ I-1 (종결) — sim-to-real 갭, IMU 내부 상관 위배 교정

교차모달 독립(ECG↔IMU): PTT-PPG로 검증 → 지지됨 (|r|<0.31).
IMU 내부 상관 위배: smv_std↔act_energy 실 0.90 vs 합성 0.33 → MVN 교정 → impact +0.027.
3-way 비교(독립/MVN/bootstrap)로 게이트 라우팅 결론 불변 확인.
SpO2 prior: Harespod 재앵커링으로 수렴 검증(nadir 84%≈85%, mean 89%≈88.3%).
남은 미검증: 응급 교차모달 결합(낙상↔빈맥 타이밍) — 정직한 한계 명시.

### ✅ I-1 (구버전) — sim-to-real 갭

ECG 임베딩은 실 P1 출력으로 교체 완료(I-2 해결). IMU는 PTT-PPG/SisFall 실데이터로
통일 보정 완료. SpO2만 합성(임상 문헌 기반) 유지. 조건부 독립 과분리는 측정노이즈+hard
case로 천장 0.95 유지 중.

### ✅ I-2 — ECG 임베딩 실데이터화 (2026-05-30)

P1 캐시(`build_p1_cache.py`)로 CPSC 실 ECG 임베딩+점수 추출 → 조립기 ECG 채널 교체 완료.
클래스 2(심혈관)=CPSC 비정상 ECG, 0/1/3/4=NSR. emb-only 분류기 cardiac recall 0.95.

### ✅ IMU 전처리 통일 보정 (2026-05-30)

fs·윈도우 불일치 + 스케일 불일치 해결. 200Hz·3초 통일, 5클래스 실데이터 재보정.
상세: `01_design_decisions.md §4`.

### ✅ Pre-flight — 데이터 전략 확정 (2026-05-30)

세 모달리티 동시+응급 레이블 공개 데이터셋 부재 확정(SHIMMER 포함 정밀 조사).
→ 클래스 조건부 조립(방법 A) + 시뮬레이터(B) + modality-dropout(C) 4단 레시피 채택.
상세: `records/01_design_decisions.md §1`.
