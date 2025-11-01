import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from rss_da.config import Config
from rss_da.data.dataset import RssDoADataset, collate_samples
from rss_da.data.synth_generator import SynthConfig, build_samples, generate_synth_samples
from rss_da.training.stage1 import Stage1Trainer


def _build_loader(cfg: Config) -> DataLoader:
    _, arrays = generate_synth_samples(SynthConfig(num_samples=128))
    samples = list(build_samples(arrays))
    dataset = RssDoADataset(samples, stage="1", modality_dropout_p=0.0, training=True)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, collate_fn=collate_samples)
    return loader


def test_forward_mix_decreases():
    cfg = Config()
    cfg.train.batch_size = 16
    trainer = Stage1Trainer(cfg)
    loader = _build_loader(cfg)
    iterator = iter(loader)
    mix_losses = []
    for step in range(10):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        outputs = trainer.train_step(batch)
        mix_losses.append(outputs.recon_mix_norm)
    assert min(mix_losses) <= mix_losses[0]
