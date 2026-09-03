"""Attention masks for channel-dependent supervised baselines."""

from __future__ import annotations

import torch
import torch.nn as nn


class _KeyMask:
    __slots__ = ("mask",)

    def __init__(self, mask: torch.Tensor):
        self.mask = mask


def _enable_attention_masks(module: nn.Module, child_name: str | None = None) -> int:
    roots = [module]
    if child_name is not None:
        roots = [
            getattr(item, child_name)
            for item in module.modules()
            if hasattr(item, child_name)
        ]
    count = 0
    for root in roots:
        for item in root.modules():
            if hasattr(item, "mask_flag"):
                item.mask_flag = True
                count += 1
    return count


def _key_mask(
    present: torch.Tensor | None,
    n_tokens: int,
    like: torch.Tensor,
) -> _KeyMask:
    if present is None:
        drop = torch.zeros(
            like.shape[0], n_tokens, dtype=torch.bool, device=like.device
        )
    else:
        drop = ~present
        if n_tokens > drop.shape[1]:
            padding = torch.zeros(
                drop.shape[0],
                n_tokens - drop.shape[1],
                dtype=torch.bool,
                device=drop.device,
            )
            drop = torch.cat((drop, padding), dim=1)
    return _KeyMask(drop[:, None, None, :])


class MaskedITransformer(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.n_attention_layers = _enable_attention_masks(base.encoder)
        if not self.n_attention_layers:
            raise TypeError("iTransformer encoder exposes no maskable attention")

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        present: torch.Tensor | None = None,
    ):
        base = self.base
        if base.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        n_channels = x_enc.shape[-1]
        encoded = base.enc_embedding(x_enc, x_mark_enc)
        encoded, _ = base.encoder(
            encoded,
            attn_mask=_key_mask(present, encoded.shape[1], encoded),
        )
        prediction = base.projector(encoded).permute(0, 2, 1)[:, :, :n_channels]

        if base.use_norm:
            prediction = prediction * stdev[:, 0, :].unsqueeze(1)
            prediction = prediction + means[:, 0, :].unsqueeze(1)
        return prediction[:, -base.pred_len :, :]


class MaskedTimeXer(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.n_attention_layers = _enable_attention_masks(
            base.encoder, child_name="cross_attention"
        )
        if not self.n_attention_layers:
            raise TypeError("TimeXer encoder exposes no maskable cross-attention")

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        present: torch.Tensor | None = None,
    ):
        base = self.base
        if base.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        endogenous, n_variables = base.en_embedding(x_enc.permute(0, 2, 1))
        exogenous = base.ex_embedding(x_enc, x_mark_enc)
        encoded = base.encoder(
            endogenous,
            exogenous,
            cross_mask=_key_mask(present, exogenous.shape[1], exogenous),
        )
        encoded = torch.reshape(
            encoded,
            (-1, n_variables, encoded.shape[-2], encoded.shape[-1]),
        )
        prediction = base.head(encoded.permute(0, 1, 3, 2)).permute(0, 2, 1)

        if base.use_norm:
            prediction = prediction * stdev[:, 0, :].unsqueeze(1)
            prediction = prediction + means[:, 0, :].unsqueeze(1)
        return prediction[:, -base.pred_len :, :]


WRAPPERS = {
    "itransformer": MaskedITransformer,
    "timexer": MaskedTimeXer,
}


def wrap(name: str, model: nn.Module) -> nn.Module:
    try:
        wrapper = WRAPPERS[name]
    except KeyError as error:
        raise ValueError(f"no channel mask is defined for {name}") from error
    return wrapper(model)
