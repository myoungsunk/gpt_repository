import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from rss_da.config import Config
from rss_da.data.dataset import RssDoADataset, collate_samples
from rss_da.data.synth_generator import SynthConfig, build_samples, generate_synth_samples
from rss_da.training.stage1 import Stage1Trainer


def test_stage1_single_step():
    cfg = Config()
    cfg.train.batch_size = 8
    trainer = Stage1Trainer(cfg)
    _, arrays = generate_synth_samples(SynthConfig(num_samples=32))
    samples = list(build_samples(arrays))
    dataset = RssDoADataset(samples, stage="1", modality_dropout_p=0.0, training=True)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, collate_fn=collate_samples)
    batch = next(iter(loader))
    outputs = trainer.train_step(batch)
    assert outputs.loss_total > 0
    assert outputs.sup0_nll >= 0
    assert outputs.sup1_nll >= 0
