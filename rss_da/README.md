# RSS-DA Stage-1→Stage-2.5

RSS 기반 DoA 회귀 도메인 적응 파이프라인으로 런타임 입력은 [z5d, φ]에 통일된다.
Two-Pass 추론을 통해 Pass-0(z5d)과 Pass-1([z5d, φ])를 순차적으로 수행한다.

## Quickstart
```bash
python -m rss_da.data.synth_generator --out data/synth.json --num-samples 512
```
