from typing import Optional

import torch
import torch.nn as nn
import yaml


class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
		pe = torch.zeros(max_len, d_model)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe, persistent=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (batch, seq_len, d_model)
		seq_len = x.size(1)
		x = x + self.pe[:seq_len].unsqueeze(0).to(x.dtype)
		return self.dropout(x)


class SmallTokenEncoder(nn.Module):
	def __init__(self, input_dim: int, proj_dim: int = 768, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
		super().__init__()
		self.proj = nn.Linear(input_dim, proj_dim)
		self.pe = PositionalEncoding(proj_dim, dropout=dropout)
		encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim, nhead=nhead, dim_feedforward=proj_dim * 4, dropout=dropout, batch_first=False)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.head = nn.Linear(proj_dim, 1)  # per-token logit

	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		# x: (batch, seq_len, input_dim)
		x = self.proj(x)  # (B, L, D)
		x = self.pe(x)
		# Transformer expects (L, B, D)
		x_t = x.permute(1, 0, 2)
		# src_key_padding_mask: (B, L) with True for pads → invert from our mask (1 for valid)
		src_key_padding_mask = None
		if mask is not None:
			src_key_padding_mask = (mask == 0)
		enc = self.encoder(x_t, src_key_padding_mask=src_key_padding_mask)
		enc = enc.permute(1, 0, 2)  # (B, L, D)
		logits = self.head(enc).squeeze(-1)  # (B, L)
		if mask is not None:
			min_val = -1e4 if logits.dtype in (torch.float16, torch.bfloat16) else -1e9
			logits = logits.masked_fill(mask == 0, min_val)
		return logits


def build_small_encoder(concat_layers: int, base_hidden: int, config_path: str = 'config.yaml') -> SmallTokenEncoder:
	with open(config_path, 'r', encoding='utf-8') as f:
		cfg = yaml.safe_load(f)
	proj_dim = int(cfg['features'].get('projection_dim', 768))
	input_dim = concat_layers * base_hidden
	model = SmallTokenEncoder(input_dim=input_dim, proj_dim=proj_dim)
	return model


if __name__ == '__main__':
	pass
