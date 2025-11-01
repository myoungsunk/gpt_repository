import pytest

torch = pytest.importorskip("torch")

from rss_da.config import Config
from rss_da.models.decoder import DecoderD
from rss_da.models.encoders import Adapter, E4, E5, Fuse
from rss_da.models.m2 import DoAPredictor
from rss_da.models.m3 import ResidualCalibrator
from rss_da.physics.combine import combine_r4_to_c


def test_model_shapes():
    cfg = Config()
    latent = cfg.train.latent_dim
    phi_dim = cfg.train.phi_dim
    e4 = E4(latent_dim=latent)
    e5 = E5(latent_dim=latent)
    fuse = Fuse(latent_dim=latent)
    adapter = Adapter(latent_dim=latent, phi_dim=phi_dim)
    decoder = DecoderD(latent_dim=latent)
    m2 = DoAPredictor(phi_dim=phi_dim, latent_dim=latent)
    batch = 4
    z5d = torch.randn(batch, 5)
    four_rss = torch.randn(batch, 4)
    h5 = e5(z5d)
    h4 = e4(four_rss)
    mask = torch.zeros(batch, 1)
    fused = fuse(h5, h4, mask)
    phi = adapter(fused)
    mu0, kappa0, _ = m2(z5d)
    assert h5.shape == (batch, latent)
    assert h4.shape == (batch, latent)
    assert fused.shape == (batch, latent)
    assert phi.shape == (batch, phi_dim)
    mu0 = mu0 if mu0.ndim == 2 else mu0[:, 0, :]
    kappa0 = kappa0 if kappa0.ndim == 2 else kappa0[:, 0, :]
    assert mu0.shape == (batch, 2)
    assert kappa0.shape == (batch, 2)
    r4_hat = decoder(h5, mu0)
    assert r4_hat.shape == (batch, 4)
    combined = combine_r4_to_c(r4_hat)
    assert combined.shape == (batch, 2)
    calibrator = ResidualCalibrator(feature_dim=5 + phi_dim)
    residual = calibrator(mu0, kappa0, torch.cat([z5d, phi], dim=-1))
    assert residual.shape == (batch, 2)
