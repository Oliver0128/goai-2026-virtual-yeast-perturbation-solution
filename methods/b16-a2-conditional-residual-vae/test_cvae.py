from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("b16_a2_core_test", HERE / "core.py")
assert SPEC is not None and SPEC.loader is not None
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


def make_model(output_dim: int = 17) -> torch.nn.Module:
    return CORE.ConditionalResidualVAE(
        context_dim=13,
        output_dim=output_dim,
        latent_dim=64,
        residual_hidden_dim=11,
        distribution_hidden_dim=19,
        decoder_hidden_dim=23,
        dropout=0.0,
        correction_bound=0.75,
    )


def test_zero_initialized_prior_mean_is_exact_dense_bypass() -> None:
    model = make_model()
    correction, audit = model.forward_prior_mean(torch.randn(5, 13))
    torch.testing.assert_close(correction, torch.zeros_like(correction))
    assert correction.abs().max().item() <= 0.75
    assert audit["p_mu"].shape == (5, 64)
    assert torch.count_nonzero(model.output_head.weight) == 0


def test_posterior_backward_and_kl_are_finite() -> None:
    CORE.set_reproducible_seed(42)
    model = make_model(output_dim=9)
    context = torch.randn(7, 13)
    residual = torch.randn(7, 9)
    mask = torch.ones(7, 9)
    mask[0, 1] = 0.0
    correction, audit = model.forward_posterior(context, residual, mask)
    prediction = correction
    absolute = CORE.masked_mse(prediction, residual, mask)
    kl_dimensions = CORE.diagonal_gaussian_kl(
        audit["q_mu"], audit["q_logvar"], audit["p_mu"], audit["p_logvar"]
    )
    kl = CORE.free_bits_kl(kl_dimensions, 0.02)
    loss = absolute + 0.002 * kl
    loss.backward()
    assert torch.isfinite(loss)
    assert model.output_head.weight.grad is not None
    assert model.prior_mu.weight.grad is not None
    assert model.posterior_mu.weight.grad is not None


def test_identical_gaussians_have_zero_raw_kl() -> None:
    mu = torch.randn(4, 64)
    logvar = torch.randn(4, 64).clamp(-2, 2)
    kl = CORE.diagonal_gaussian_kl(mu, logvar, mu, logvar)
    torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-6, rtol=0)


def test_inference_api_cannot_receive_target_proteins() -> None:
    parameters = list(
        inspect.signature(CORE.ConditionalResidualVAE.forward_prior_mean).parameters
    )
    assert parameters == ["self", "context"]


def test_actual_dimension_parameter_budget() -> None:
    model = CORE.ConditionalResidualVAE(
        context_dim=256,
        output_dim=4422,
        latent_dim=64,
        residual_hidden_dim=96,
        distribution_hidden_dim=128,
        decoder_hidden_dim=128,
        dropout=0.1,
        correction_bound=0.75,
    )
    assert CORE.trainable_parameter_count(model) <= 1_500_000


def test_train_only_pairing_excludes_unmatched_treatment() -> None:
    index = pd.Index(["c_w", "c_d", "t_ok", "t_no"], name="sample_ID")
    metadata = pd.DataFrame(
        {
            "perturbation_no_concentration": ["Water", "DMSO", "Drug", "Drug"],
            "key": ["A", "A", "A", "B"],
        },
        index=index,
    )
    truth = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 8.0], [9.0, 9.0]],
        index=index,
    )
    drug = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    paired = CORE.build_train_only_paired_delta(
        metadata, truth, drug, ("key",), ("Water", "DMSO"),
        "perturbation_no_concentration",
    )
    assert paired.manifest["matched_treatment_rows"] == 1
    assert paired.manifest["unmatched_treatment_rows"] == 1
    np.testing.assert_allclose(paired.target_delta[2], [3.0, 5.0])
    np.testing.assert_allclose(paired.control_weights[2], [0.5, 0.5])


def test_frozen_e7_source_exists_and_is_hashable() -> None:
    assert CORE.E7_CORE_PATH.is_file()
    assert len(CORE.frozen_e7_core_sha256()) == 64
