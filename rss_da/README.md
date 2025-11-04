# RSS-DA Stage-1→Stage-2.5

RSS 기반 DoA 회귀 도메인 적응 파이프라인으로 런타임 입력은 [z5d, φ]에 통일된다.
Two-Pass 추론을 통해 Pass-0(z5d)과 Pass-1([z5d, φ])를 순차적으로 수행한다.

## Quickstart
```bash
python -m rss_da.data.synth_generator --out data/synth.json --num-samples 512
```

## VS Code 작업 명령
Visual Studio Code에서 φ 게이팅 및 디코더 성능 진단을 빠르게 재현할 수 있도록 두 개의 Task를 추가했습니다. 명령 팔레트에서 **"Tasks: Run Task"**를 실행한 뒤 아래 태스크를 선택하면 됩니다.

1. **Stage-1 M2 φ-gate diagnostics** – Pass-0/1 파이프라인을 실행하면서 디코더 4RSS 재구성 오차, φ 게이트 통계, forward consistency 로그를 남깁니다. φ 게이트 분위수(`--phi_gate_quantile=0.6`)와 최소 유지 비율(`--phi_gate_min_keep=0.2`)이 적용됩니다.
2. **Stage-1 φ-gated M3 finetuning** – `phase3` 프리셋을 사용해 M3를 활성화한 미세조정을 수행합니다. 동일한 φ 게이팅 설정을 사용하며, M3 게이트 및 품질 관련 로그를 확인할 수 있습니다.

태스크 정의는 [`.vscode/tasks.json`](../.vscode/tasks.json)에 있으며, 필요한 경우 원하는 하이퍼파라미터나 로그 경로(`runs/vscode_*`)를 조정해 사용할 수 있습니다.
