#
# Code is adapted from https://github.com/lucidrains/e2-tts-pytorch
#

"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList, Sequential, Linear
import torch.nn.functional as F

from torchdiffeq import odeint
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, pack, unpack
from x_transformers import Attention, FeedForward, RMSNorm, AdaptiveRMSNorm
from x_transformers.x_transformers import RotaryEmbedding
from gateloop_transformer import SimpleGateLoopLayer

from .tensor_typing import Float


class Identity(Module):
    def forward(self, x, **kwargs):
        return x


class AdaLNZero(Module):
    def __init__(
        self,
        dim: int,
        dim_condition: Optional[int] = None,
        init_bias_value: float = -2.0,
    ):
        super().__init__()
        dim_condition = dim_condition or dim
        self.to_gamma = nn.Linear(dim_condition, dim)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def forward(self, x: torch.Tensor, *, condition: torch.Tensor) -> torch.Tensor:
        # condition shape: (b, d) or (b, 1, d)
        if condition.ndim == 2:
            condition = rearrange(condition, "b d -> b 1 d")
        gamma = self.to_gamma(condition).sigmoid()
        return x * gamma


def exists(v: Any) -> bool:
    return v is not None


def default(v: Any, d: Any) -> Any:
    return v if exists(v) else d


def divisible_by(num: int, den: int) -> bool:
    return (num % den) == 0


class Transformer(Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        cond_on_time: bool = True,
        skip_connect_type: str = "concat",
        abs_pos_emb: bool = True,
        max_seq_len: int = 8192,
        heads: int = 8,
        dim_head: int = 64,
        num_gateloop_layers: int = 1,
        dropout: float = 0.1,
        num_registers: int = 32,
        attn_kwargs: Dict[str, Any] = dict(
            gate_value_heads=True, softclamp_logits=True
        ),
        ff_kwargs: Dict[str, Any] = dict(),
    ):
        super().__init__()
        assert divisible_by(depth, 2), "depth needs to be even"

        self.max_seq_len = max_seq_len
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if abs_pos_emb else None
        self.dim = dim
        self.skip_connect_type = skip_connect_type
        needs_skip_proj = skip_connect_type == "concat"
        self.depth = depth
        self.layers = ModuleList([])

        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.zeros(num_registers, dim))
        nn.init.normal_(self.registers, std=0.02)

        self.rotary_emb = RotaryEmbedding(dim_head)
        self.cond_on_time = cond_on_time
        rmsnorm_klass = AdaptiveRMSNorm if cond_on_time else RMSNorm
        postbranch_klass = partial(AdaLNZero, dim=dim) if cond_on_time else Identity

        self.time_cond_mlp = (
            Sequential(Rearrange("... -> ... 1"), Linear(1, dim), nn.SiLU())
            if cond_on_time
            else nn.Identity()
        )

        for ind in range(depth):
            is_later_half = ind >= (depth // 2)
            gateloop = SimpleGateLoopLayer(dim=dim)
            attn_norm = rmsnorm_klass(dim)
            attn = Attention(
                dim=dim, heads=heads, dim_head=dim_head, dropout=dropout, **attn_kwargs
            )
            attn_adaln_zero = postbranch_klass()
            ff_norm = rmsnorm_klass(dim)
            ff = FeedForward(dim=dim, glu=True, dropout=dropout, **ff_kwargs)
            ff_adaln_zero = postbranch_klass()
            skip_proj = (
                Linear(dim * 2, dim, bias=False)
                if needs_skip_proj and is_later_half
                else None
            )

            self.layers.append(
                ModuleList(
                    [
                        gateloop,
                        skip_proj,
                        attn_norm,
                        attn,
                        attn_adaln_zero,
                        ff_norm,
                        ff,
                        ff_adaln_zero,
                    ]
                )
            )

        self.final_norm = RMSNorm(dim)

    def forward(
        self,
        x: Float["b n d"],
        times: Optional[Float["b"] | Float[""]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (b, n, d)
            times: (b,) or scalar if cond_on_time is True
            mask: (b, n) boolean or 0/1 mask for attention
        """
        b, n, device = x.shape[0], x.shape[1], x.device
        assert not (
            exists(times) ^ self.cond_on_time
        ), "`times` must be passed in if `cond_on_time` is set to `True`, and vice versa."

        norm_kwargs = {}

        # Absolute positional embedding
        if exists(self.abs_pos_emb):
            # you may want to guard for n <= self.max_seq_len
            pos_indices = torch.arange(n, device=device)
            x = x + self.abs_pos_emb(pos_indices)

        # Time conditioning
        if exists(times):
            if times.ndim == 0:
                times = repeat(times, " -> b", b=b)
            times = self.time_cond_mlp(times)  # (b, d) or (b, 1, d)
            norm_kwargs["condition"] = times

        # Concat registers to the sequence
        registers = repeat(self.registers, "r d -> b r d", b=b)
        x, registers_packed_shape = pack((registers, x), "b * d")

        # Build the rotary embeddings for this sequence length
        rotary_pos_emb = self.rotary_emb.forward_from_seq_len(x.shape[-2])

        # Similarly extend the mask to registers + real tokens if given
        if mask is not None:
            # mask: (b, n), we have total length = r + n after packing
            # The first `r` (num_registers) are never "masked" out
            # so we build a new mask of shape (b, r + n)
            reg_mask = x.new_ones(b, self.num_registers, dtype=mask.dtype)
            mask = torch.cat([reg_mask, mask], dim=1)  # (b, r + n)

        # We'll keep track of skip connections
        skips = []

        for ind, (
            gateloop,
            maybe_skip_proj,
            attn_norm,
            attn,
            maybe_attn_adaln_zero,
            ff_norm,
            ff,
            maybe_ff_adaln_zero,
        ) in enumerate(self.layers):

            layer_idx = ind + 1
            is_first_half = layer_idx <= (self.depth // 2)

            # If in the first half, push x onto skip stack
            if is_first_half:
                skips.append(x)
            else:
                # Retrieve matching skip
                skip = skips.pop()
                if self.skip_connect_type == "concat":
                    x = torch.cat((x, skip), dim=-1)
                    x = maybe_skip_proj(x)

            # GateLoop
            x = gateloop(x) + x

            # Attention
            attn_out = attn(
                attn_norm(x, **norm_kwargs),
                rotary_pos_emb=rotary_pos_emb,
                mask=mask,  # pass mask here
            )
            x = x + maybe_attn_adaln_zero(attn_out, **norm_kwargs)

            # Feed-forward
            ff_out = ff(ff_norm(x, **norm_kwargs))
            x = x + maybe_ff_adaln_zero(ff_out, **norm_kwargs)

        assert len(skips) == 0, "Skip-connection stack not empty at the end!"

        # Unpack back
        _, x = unpack(x, registers_packed_shape, "b * d")

        return self.final_norm(x)


class VoiceRestore(nn.Module):
    def __init__(
        self,
        sigma: float = 0.0,
        transformer: Optional[Dict[str, Any]] = None,
        odeint_kwargs: Optional[Dict[str, Any]] = None,
        num_channels: int = 100,
    ):
        super().__init__()
        self.sigma = sigma
        self.num_channels = num_channels

        # For simplicity, always cond_on_time = True in transformer config
        self.transformer = Transformer(**transformer, cond_on_time=True)

        # Default ODE integration settings
        self.odeint_kwargs = odeint_kwargs or {
            "atol": 1e-5,
            "rtol": 1e-5,
            "method": "midpoint",
        }

        self.proj_in = nn.Linear(num_channels, self.transformer.dim)
        self.cond_proj = nn.Linear(num_channels, self.transformer.dim)
        self.to_pred = nn.Linear(self.transformer.dim, num_channels)

    def transformer_with_pred_head(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Projects input x, optionally adds condition, feeds through Transformer,
        and returns final prediction in dimension of num_channels.
        """
        x = self.proj_in(x)  # (b, n, dim)
        if cond is not None:
            cond_proj = self.cond_proj(cond)
            x = x + cond_proj  # broadcast if shapes match suitably

        attended = self.transformer(x, times=times, mask=mask)
        return self.to_pred(attended)  # (b, n, num_channels)

    def cfg_transformer_with_pred_head(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cfg_strength: float = 0.5,
    ):
        """
        Classifier-free guidance variant of the transformer forward.
        If cfg_strength <= 0, effectively the normal forward pass.
        """
        pred = self.transformer_with_pred_head(x, times=times, cond=cond, mask=mask)

        if cfg_strength < 1e-5:
            # no guidance
            return pred if mask is None else pred * mask.unsqueeze(-1)

        # null (no condition) pass
        null_pred = self.transformer_with_pred_head(
            x, times=times, cond=None, mask=mask
        )

        guided = pred + (pred - null_pred) * cfg_strength
        return guided if mask is None else guided * mask.unsqueeze(-1)

    @torch.no_grad()
    def sample(
        self,
        processed: torch.Tensor,
        steps: int = 32,
        cfg_strength: float = 0.5,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Example sampling routine using an ODE solver.  For many text/audio models,
        you'll use something else, but this shows how you might incorporate the
        same forward pass + mask into an ODE integration.
        """
        self.eval()
        # times from 0 -> 1
        times = torch.linspace(0, 1, steps, device=processed.device)

        def ode_fn(t: torch.Tensor, x: torch.Tensor):
            return self.cfg_transformer_with_pred_head(
                x, times=t, cond=processed, mask=mask, cfg_strength=cfg_strength
            )

        # Starting from noise
        y0 = torch.randn_like(processed)
        trajectory = odeint(ode_fn, y0, times, **self.odeint_kwargs)
        restored = trajectory[-1]
        return restored
