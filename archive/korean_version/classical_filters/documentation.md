# EventGuard: 하드웨어 효율성 비교 (Paper Table 1 기준)

**해상도**: 1280x720 (N = 921,600 pixels) | **이벤트 레이트**: 5 MEPS

---

## 전체 비교 테이블

| Filter | Mem(#) | Op/event | Type | Energy/Event | Ops/Sec |
|:---|:---:|---:|:---:|---:|---:|
| BAF [13] | N² | 11 | Int | 0.11 pJ | 55 M |
| **ONF** | **4N** | **8** | **Int** | **0.08 pJ** | **40 M** |
| STCF | N² | 25 | Int | 0.25 pJ | 125 M |
| NN Filter | N² | 40 | Int | 0.40 pJ | 200 M |
| **v8 (Sparse)** | **N² + 0.7KB** | **≈660** | **Binary** | **≈20 pJ** | **3.3 G** ⭐ |
| MLPF (98-20-1) | N² + MLP | ≈2k | INT4 | ≈400 pJ | 10 G |
| v8 (Dense) | 14MB | ≈13k | Binary | ≈400 pJ | 66 G |
| EDnCNN [30] | >48M | 167M | FP32 | 768 mJ | 835 T |
| **AEDNet** | **>100M** | **~110M** | **FP32** | **~506 mJ** | **550 T** |

**에너지 계산 기준** (Horowitz 2014, 45nm):
- Int: 0.01 pJ/Op
- Binary: 0.03 pJ/Op
- INT4: 0.2 pJ/Op
- FP32: 4.6 pJ/Op

---

## v8_bconvsnn Ops/Event 일반화 공식

### Sparse Mode (Event-Driven)

```
Ops/Event = Σ (C_in × C_out × K²) × Sparsity
```

**우리 아키텍처**: [1→16→32→2], K=3, Sparsity=0.1 (10%)

| Layer | C_in | C_out | K² | Sparsity | Ops |
|:---|:---:|:---:|:---:|:---:|---:|
| L1 (SNN) | 1 | 16 | 9 | 1.0 | **144** |
| L2 (BNN) | 16 | 32 | 9 | 0.1 | **461** |
| L3 (BNN) | 32 | 2 | 9 | 0.1 | **58** |
| **Total** | - | - | - | - | **≈663** |

### 일반화 공식

```
Ops/Event = C₁×K² + (C₁×C₂ + C₂×C₃) × K² × S
```

---

## 효율성 비교 (v8 Sparse 기준)

| 비교 대상 | Ops 배율 | Energy 배율 |
|:---|---:|---:|
| vs **AEDNet** | **166,667x** 적음 | **25,300,000x** 적음 |
| vs **EDnCNN** | **252,727x** 적음 | **38,400,000x** 적음 |
| vs **MLPF** | **3x** 적음 | **20x** 적음 |
| vs **ONF** | 82x 많음 | 250x 많음 |
| vs **BAF** | 60x 많음 | 182x 많음 |

---

## 결론

**v8_bconvsnn (Sparse)**:
- **Ops/Event** = `O(C² × K² × S)` ≈ **660 Ops** (해상도 무관)
- **Energy/Event** ≈ **20 pJ** (Binary 연산)
- **Memory** = `O(N²)` + 0.7KB weights (BAF와 동급)
- **장점**: MLPF보다 3배 효율적이면서 Full HD 처리 가능
