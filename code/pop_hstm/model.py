from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_STATIC_FILES
from .utils import compute_group_count

def build_model_input_channels(static_files: Sequence[str] = DEFAULT_STATIC_FILES) -> int:
    """Return the total input channel count used by Pop-HSTM."""
    return 1 + len(static_files) + 5 + 1

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution used by the DS blocks."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

class ReflectionPad2dTime(nn.Module):
    """Reflection padding that accepts 4D or 5D tensors."""

    def __init__(self, pad: tuple[int, int]) -> None:
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ph, pw = self.pad
        pad = (pw, pw, ph, ph)
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            reshaped = x.reshape(b * t, c, h, w)
            padded = F.pad(reshaped, pad, mode="reflect")
            new_h, new_w = padded.shape[-2:]
            return padded.view(b, t, c, new_h, new_w)
        if x.dim() == 4:
            return F.pad(x, pad, mode="reflect")
        raise ValueError("ReflectionPad2dTime expects 4D or 5D tensors.")

class SinePositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding."""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x shape: (seq_len, batch, d_model)."""
        seq_len = x.size(0)
        return x + self.pe[:seq_len]

class ResidualRefiner(nn.Module):
    """
    Lightweight 2D residual CNN used for fine-grained post-refinement of the main output.
    The design follows the idea of stacked 3x3 convolutions plus local residual correction.
    """

    def __init__(self, in_channels: int = 1, hidden: int = 32) -> None:
        super().__init__()
        k = 3
        p = k // 2
        self.conv1 = nn.Conv2d(in_channels, hidden, k, padding=p)
        self.conv2 = nn.Conv2d(hidden, hidden, k, padding=p)
        self.conv3 = nn.Conv2d(hidden, hidden, k, padding=p)
        self.conv4 = nn.Conv2d(hidden, hidden, k, padding=p)
        self.conv5 = nn.Conv2d(hidden, hidden, k, padding=p)
        self.conv6 = nn.Conv2d(hidden, hidden, k, padding=p)
        self.conv_out = nn.Conv2d(hidden, 1, k, padding=p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        c1 = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(c1))
        r1 = c2 + c1

        c3 = self.relu(self.conv3(r1))
        c4 = self.relu(self.conv4(c3))
        r2 = c4 + r1

        c5 = self.relu(self.conv5(r2))
        c6 = self.relu(self.conv6(c5))
        r3 = c6 + r2

        return self.conv_out(self.relu(r3))

class TemporalTransformer(nn.Module):
    """Applies Transformer encoder over temporal dimension per spatial location."""

    def __init__(
            self,
            feature_dim: int,
            hidden_dim: int,
            num_heads: int,
            num_layers: int,
            dropout: float = 0.1,
            max_seq_len: int = 168,  # Increase to cover one full week of hourly steps.
    ) -> None:
        super().__init__()
        self.proj_in = nn.Linear(feature_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, feature_dim)
        # Ensure hidden_dim is divisible by num_heads.
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = SinePositionalEncoding(hidden_dim, max_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        seq = x.permute(0, 3, 4, 1, 2).contiguous().view(b * h * w, t, c)
        seq = self.proj_in(seq)
        seq = seq.permute(1, 0, 2)
        seq = self.pos_encoding(seq)
        encoded = self.encoder(seq)
        encoded = encoded.permute(1, 0, 2)
        encoded = self.proj_out(encoded).view(b, h, w, t, c)
        return encoded.permute(0, 3, 4, 1, 2)

class ConvLSTMCell(nn.Module):
    """ConvLSTM cell without builtin padding (padding handled outside)."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = 0
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
            self,
            input_tensor: torch.Tensor,
            cur_state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = cur_state
        if input_tensor.shape[-2:] != h_cur.shape[-2:]:
            diff_h = input_tensor.shape[-2] - h_cur.shape[-2]
            diff_w = input_tensor.shape[-1] - h_cur.shape[-1]
            if diff_h < 0 or diff_w < 0:
                raise ValueError(
                    "Hidden state spatial dims larger than input; cannot align tensors."
                )
            pad_top = diff_h // 2
            pad_bottom = diff_h - pad_top
            pad_left = diff_w // 2
            pad_right = diff_w - pad_left
            h_cur = F.pad(h_cur, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(combined_conv, chunks=4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(
            self, batch_size: int, spatial_size: tuple[int, int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c

class ConvLSTMLayer(nn.Module):
    """Single layer ConvLSTM that can optionally return sequences."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            kernel_size: int,
            return_sequences: bool,
    ) -> None:
        super().__init__()
        self.return_sequences = return_sequences
        self.kernel_size = kernel_size
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        batch, seq_len, _, height, width = x.shape
        out_height = height - self.kernel_size + 1
        out_width = width - self.kernel_size + 1
        h, c = self.cell.init_hidden(batch, (out_height, out_width), x.device)

        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t], (h, c))
            if self.return_sequences:
                outputs.append(h)
        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        return h

class DSConvLSTMNet(nn.Module):
    """PyTorch implementation of the DS-ConvLSTM architecture."""

    def __init__(
            self,
            in_channels: int,
            kn_s: int = 3,
            dsfilter1: int = 8,
            dsfilter2: int = 4,
            filter1: int = 30,
            filter2: int = 20,
            unit1: int = 4,
            use_transformer: bool = True,
            transformer_dim: int = 64,
            transformer_heads: int = 4,
            transformer_layers: int = 2,
            transformer_dropout: float = 0.1,
            use_residual_refiner: bool = True,
            residual_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.kernel = kn_s
        self.pad = ReflectionPad2dTime((kn_s // 2, kn_s // 2))
        self.use_transformer = use_transformer
        self.use_residual_refiner = use_residual_refiner
        self.ds_conv1 = DepthwiseSeparableConv2d(in_channels, dsfilter1, kn_s)
        self.ds_norm1 = nn.GroupNorm(compute_group_count(dsfilter1), dsfilter1)
        self.ds_conv2 = DepthwiseSeparableConv2d(dsfilter1, dsfilter2, kn_s)
        self.ds_norm2 = nn.GroupNorm(compute_group_count(dsfilter2), dsfilter2)
        self.ds_conv3 = nn.Conv2d(dsfilter2, 1, kernel_size=kn_s, bias=True)
        self.skip_reduce = nn.Conv2d(1, unit1, kernel_size=1, bias=False)
        self.skip_norm = nn.GroupNorm(compute_group_count(unit1), unit1)
        self.skip_out = nn.Conv2d(unit1, 1, kernel_size=3, padding=1, bias=True)

        self.convlstm1 = ConvLSTMLayer(1, filter1, kn_s, return_sequences=True)
        self.norm_lstm1 = nn.GroupNorm(compute_group_count(filter1), filter1)
        if self.use_transformer:
            self.temporal_transformer = TemporalTransformer(
                feature_dim=filter1,
                hidden_dim=transformer_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=transformer_dropout,
                max_seq_len=168,  # Increase maximum sequence length

            )
        self.convlstm2 = ConvLSTMLayer(filter1, filter2, kn_s, return_sequences=False)
        self.norm_lstm2 = nn.GroupNorm(compute_group_count(filter2), filter2)
        self.pointwise = nn.Conv2d(filter2, unit1, kernel_size=1)
        self.dropout2d = nn.Dropout2d(p=0.1)
        self.final_conv = nn.Conv2d(unit1, 1, kernel_size=kn_s)
        self.fusion_refine = nn.Sequential(
            nn.Conv2d(2, unit1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(compute_group_count(unit1), unit1),
            nn.ReLU(inplace=True),
            nn.Conv2d(unit1, 1, kernel_size=3, padding=1, bias=True),
        )
        self.residual_refiner = ResidualRefiner(in_channels=1,
                                                hidden=residual_hidden) if self.use_residual_refiner else None

    @staticmethod
    def _match_spatial(src: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        """Align spatial size of src tensor to target height/width via crop/pad."""
        tgt_h, tgt_w = target_hw
        src_h, src_w = src.shape[-2:]
        diff_h = tgt_h - src_h
        diff_w = tgt_w - src_w
        if diff_h == 0 and diff_w == 0:
            return src
        if diff_h < 0:
            top = (-diff_h) // 2
            bottom = top + tgt_h
            src = src[..., top:bottom, :]
        else:
            pad_top = diff_h // 2
            pad_bottom = diff_h - pad_top
            src = F.pad(src, (0, 0, pad_top, pad_bottom), mode="reflect")
        if diff_w < 0:
            left = (-diff_w) // 2
            right = left + tgt_w
            src = src[..., :, left:right]
        else:
            pad_left = diff_w // 2
            pad_right = diff_w - pad_left
            src = F.pad(src, (pad_left, pad_right, 0, 0), mode="reflect")
        return src

    @staticmethod
    def _time_distributed(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
        batch, seq_len, channels, height, width = x.shape
        reshaped = x.view(batch * seq_len, channels, height, width)
        out = module(reshaped)
        _, out_channels, out_h, out_w = out.shape
        return out.view(batch, seq_len, out_channels, out_h, out_w)

    @staticmethod
    def _time_distributed_norm(x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        batch, seq_len, channels, height, width = x.shape
        reshaped = x.view(batch * seq_len, channels, height, width)
        normed = norm(reshaped)
        return normed.view(batch, seq_len, channels, height, width)

    @staticmethod
    def _apply_norm3d(x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        x_perm = x.permute(0, 2, 1, 3, 4)  # B, C, T, H, W
        normed = norm(x_perm)
        return normed.permute(0, 2, 1, 3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Time-distributed DS convolutions
        x = self.pad(x)
        x = self._time_distributed(x, self.ds_conv1)
        x = self._time_distributed_norm(x, self.ds_norm1)

        x = self.pad(x)
        x = self._time_distributed(x, self.ds_conv2)
        x = self._time_distributed_norm(x, self.ds_norm2)

        x = self.pad(x)
        x = self._time_distributed(x, self.ds_conv3)
        skip_map = x[:, -1]

        # ConvLSTM stack
        x = self.pad(x)
        x = self.convlstm1(x)
        if self.use_transformer:
            x = self.temporal_transformer(x)
        x = self._apply_norm3d(x, self.norm_lstm1)

        x = self.pad(x)
        x = self.convlstm2(x)
        x = self.norm_lstm2(x)

        detail_feat = self.pointwise(x)
        detail_feat = self.dropout2d(F.relu(detail_feat))
        logits = self.final_conv(self.pad(detail_feat))

        skip_map = self.pad(skip_map)
        skip_feat = self.skip_reduce(skip_map)
        skip_feat = F.relu(self.skip_norm(skip_feat))
        skip_logits = self.skip_out(skip_feat)
        skip_logits = self._match_spatial(skip_logits, logits.shape[-2:])

        fusion = torch.cat([logits, skip_logits], dim=1)
        refined = self.fusion_refine(fusion)
        out = logits + refined
        if self.residual_refiner is not None:
            # Residual refinement: make only a small correction on top of the main output.
            out = out + self.residual_refiner(out)
        return out


# -----------------------------------------------------------------------------
# Metrics, losses, training, and calibration
# -----------------------------------------------------------------------------
