# SNR 평가 스크립트 (`evaluate_snr.py`) 구현 계획

## 목표
- `DVSCLEAN_FRAME` 및 `ESD` 데이터셋에 대해 클래식 필터(BAF, STCF, ONF, STCF_Sub)의 **SNR 성능**을 평가합니다.
- 기존 ROC-AUC 코드(`evaluate_roc_auc.py`)를 건드리지 않고 별도 파일로 만듭니다.

## 주요 기능
1. **입력**: 데이터셋 디렉토리 경로 (`--data_dir`)
2. **필터**: BAF, STCF, ONF(수정된 버전), STCF_Sub
3. **파라미터**: 고정된 `tau` (예: 1/FPS) 또는 최적의 `tau` 사용. (사용자가 FPS별로 테스트하길 원하므로 `tau = 1/FPS`가 합리적임)
   - `--fps` 인자를 받아서 `tau = 1.0 / fps`로 설정.
4. **SNR 계산**:
   - `Signal` = 필터 통과한 True Signal 개수
   - `Noise` = 필터 통과한 Noise 개수 (False Positive)
   - `SNR (dB) = 20 * log10(Signal / Noise)` (보통 Noise가 0이면 Inf)
   - 또는 `SNR = Signal / Noise` (ratio)
   - 논문 정의에 따름: $SNR = \frac{N_{signal}}{N_{noise}}$ 또는 $20 \log_{10} ...$
   - 여기서는 일반적인 **20 log10** 방식을 사용하거나, 사용자가 기존에 보던 방식을 따릅니다. (보통 Ratio 자체를 보기도 함)

## 구현 상세
- `evaluate_roc_auc.py`의 필터 함수들(`apply_baf_filter` 등)을 import하거나 복사해옵니다. (ONF는 수정된 버전 복사 필수)
- 디렉토리 내 모든 `.npy` 파일을 순회하며 Signal/Noise Count를 누적합니다.
- 최종 SNR을 출력합니다.

## 실행 계획
1. `evaluate_snr.py` 작성.
2. `run_snr_eval.sh` 스크립트 작성 (각 데이터셋/FPS 설정에 맞춰 실행).
3. 실행 및 결과 보고.
