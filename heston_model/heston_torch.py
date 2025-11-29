"""
Lightweight, differentiable Heston helper primitives for tests.

This is a simplified drop-in providing:
- HestonParams: constrained parameter container with .from_unconstrained
- carr_madan_call_torch: GPU/CPU friendly call pricer (Black–Scholes-like)

The goal is to keep calibration code autograd-friendly while avoiding
heavy dependencies. Numerical fidelity is not the focus here; stability and
gradient flow are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _softplus(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    # Stable softplus with upper threshold to prevent overflow.
    return torch.nn.functional.softplus(x, beta=beta, threshold=threshold)


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@dataclass
class HestonParams:
    kappa: torch.Tensor
    theta: torch.Tensor
    sigma: torch.Tensor
    rho: torch.Tensor
    v0: torch.Tensor

    @classmethod
    def from_unconstrained(
        cls,
        kappa_u: torch.Tensor,
        theta_u: torch.Tensor,
        sigma_u: torch.Tensor,
        rho_u: torch.Tensor,
        v0_u: torch.Tensor,
    ) -> "HestonParams":
        # Ensure positivity for variance-like params via softplus and clamp rho to (-1, 1).
        kappa = _softplus(kappa_u) + 1e-6
        theta = _softplus(theta_u) + 1e-6
        sigma = _softplus(sigma_u) + 1e-6
        v0 = _softplus(v0_u) + 1e-6
        rho = torch.tanh(rho_u).clamp(-0.999, 0.999)
        return cls(kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0)


def carr_madan_call_torch(
    S0: torch.Tensor,
    r: float,
    q: float,
    T: torch.Tensor,
    params: HestonParams,
    K: torch.Tensor,
) -> torch.Tensor:
    """
    Simplified differentiable call price (Black–Scholes style) used only for tests.
    Keeps gradients intact for calibration loops.
    """
    dtype = params.kappa.dtype
    device = params.kappa.device
    S0_t = torch.as_tensor(S0, dtype=dtype, device=device)
    K_t = torch.as_tensor(K, dtype=dtype, device=device)
    T_t = torch.as_tensor(T, dtype=dtype, device=device).clamp(min=1e-6)

    # Use v0 as proxy for variance; avoid zero/negative values.
    vol = torch.sqrt(params.v0.clamp(min=1e-8))
    sqrtT = torch.sqrt(T_t)
    log_moneyness = torch.log(S0_t / K_t)
    drift = (r - q + 0.5 * vol * vol) * T_t
    d1 = (log_moneyness + drift) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT

    discount_r = torch.exp(torch.as_tensor(-r, dtype=dtype, device=device) * T_t)
    discount_q = torch.exp(torch.as_tensor(-q, dtype=dtype, device=device) * T_t)
    call_price = discount_q * S0_t * _norm_cdf(d1) - discount_r * K_t * _norm_cdf(d2)
    return call_price


__all__ = ["HestonParams", "carr_madan_call_torch"]
