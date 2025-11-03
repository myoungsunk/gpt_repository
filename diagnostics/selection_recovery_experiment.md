# Stage-1 "선택성 복구" 실험 커맨드

다음 커맨드는 Stage-1 M3 선택성 복구 실험을 재현할 때 사용한다. `rss_da/train.py` 스크립트를 Stage-1, `finetune_m3` 단계로 실행하며, 요청된 게이팅/게인 파라미터를 모두 명시한다.

```bash
python -m rss_da.train \
  --stage 1 \
  --phase finetune_m3 \
  --data_root rss_da/data/raw_data/standardized_v2 \
  --epochs 30 \
  --batch_size 32 \
  --logdir runs/selection_recovery \
  --enable_m3 \
  --m3_freeze_m2 \
  --load_m2_ckpt runs/s1_phaseA_m2/phaseA/best_m2.pth \
  --m3_gate_mode kappa \
  --m3_gate_keep_threshold 0.60 \
  --m3_quantile_keep 0.65 \
  --m3_gate_temp 1.5 \
  --m3_gain_start 0.1 \
  --m3_gain_end 0.7 \
  --m3_gain_ramp_steps 20 \
  --m3_delta_max_deg 2.0 \
  --m3_lambda_gate_entropy 5e-3 \
  --m3_lambda_resid 0.10 \
  --m3_detach_warmup_epochs 2 \
  --m3_keep_warmup_epochs 2
```

## 플래그 설명
- `--m3_gate_keep_threshold 0.60`과 `--m3_quantile_keep 0.65`는 두 규칙 중 더 엄격한 기준을 자동으로 적용하도록 함께 지정한다.
- `--m3_gain_start`, `--m3_gain_end`, `--m3_gain_ramp_steps`는 게인 램프 스케줄을 0.1 → 0.7, 20 step으로 설정한다.
- `--m3_delta_max_deg 2.0`은 Δ cap을 2°로 고정한다.
- `--m3_lambda_resid 0.10`은 잔차 규제를 요구된 값으로 조정한다.
- `--m3_detach_warmup_epochs 2`와 `--m3_keep_warmup_epochs 2`는 워밍업을 각각 detach/keep 스케줄에 2 epoch 비율로 반영한다. Stage-1 초기 로그에 `Resolved training epochs`와 `[Stage-1][M3] resolved schedule` 메시지가 출력되므로 실제 스텝/에폭 환산 값을 반드시 확인한다.
- 다중 행으로 입력할 때는 각 행 끝의 백슬래시(`\`)를 유지하거나 한 줄로 실행해 인자 누락을 방지한다.

필요에 따라 `--data_root`, `--epochs`, `--batch_size`, `--logdir`를 조정할 수 있다.
