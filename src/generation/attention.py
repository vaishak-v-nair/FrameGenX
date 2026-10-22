import math
import re
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


def find_token_indices(tokenizer, prompt: str, target_words: List[str]) -> List[int]:
    """
    Finds exact token indices for target words using character offsets.
    Assumes a Fast tokenizer is provided (e.g. CLIPTokenizerFast).
    """
    tokens = tokenizer(prompt, return_offsets_mapping=True)
    offsets = tokens.offset_mapping
    
    indices = set()
    for target in target_words:
        target_lower = target.lower()
        # Find all character offsets of the target word in the prompt
        for match in re.finditer(r'\b' + re.escape(target_lower) + r'\b', prompt.lower()):
            start, end = match.span()
            # Find tokens that overlap with this span
            for i, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == tok_end:  # Special tokens like <|startoftext|>
                    continue
                if tok_start < end and tok_end > start:
                    indices.add(i)
                    
    return sorted(list(indices))


class RegionalAttentionProcessor:
    """
    Custom Attention Processor for Diffusers SDXL UNet.
    Enforces Regional Latent Conditioning (Attention Masking) by heavily penalizing 
    cross-attention scores outside the designated bounding boxes for specific tokens.
    """
    def __init__(self, token_masks: Dict[int, List[Dict[str, float]]], resolution: int = 512):
        # token_masks maps token_index -> [{"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}, ...]
        self.token_masks = token_masks
        self.resolution = resolution

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ):
        # Self-attention -> default SDPA behavior
        if encoder_hidden_states is None:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            
            query = query.view(query.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(key.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(value.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
            
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
            
            if hasattr(attn, "to_out"):
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

        # Cross-attention -> custom masking behavior
        batch_size, sequence_length, dim = hidden_states.shape
        h = w = int(math.sqrt(sequence_length))
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Compute raw attention scores
        query = query * attn.scale
        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        # Apply spatial masks for specific tokens
        if self.token_masks:
            # We initialize a mask tensor on-demand to avoid OOM
            mask = torch.zeros((1, 1, h * w, attention_scores.shape[-1]), 
                               device=attention_scores.device, dtype=attention_scores.dtype)
            
            for token_idx, bboxes in self.token_masks.items():
                # Penalize attention outside the bounding boxes
                spatial_mask = torch.full((h, w), -10000.0, device=attention_scores.device, dtype=attention_scores.dtype)
                
                # Union of all bounding boxes for this token
                for bbox in bboxes:
                    x1 = int(bbox["xmin"] * w / self.resolution)
                    y1 = int(bbox["ymin"] * h / self.resolution)
                    x2 = int(bbox["xmax"] * w / self.resolution)
                    y2 = int(bbox["ymax"] * h / self.resolution)
                    spatial_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 0.0
                
                mask[0, 0, :, token_idx] = spatial_mask.view(-1)
            
            attention_scores = attention_scores + mask

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(query.dtype)

        # Output projection
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if hasattr(attn, "to_out"):
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            
        return hidden_states
