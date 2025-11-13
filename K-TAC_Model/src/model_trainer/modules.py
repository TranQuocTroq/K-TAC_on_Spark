# src/model_trainer/modules.py

import torch
import torch.nn as nn

class KAVTC_Module_G2(nn.Module):
    """
    SỬA LỖI ĐA GPU: Loại bỏ vòng lặp 'for', thay bằng phép nhân ma trận.
    """
    def __init__(self, embed_dim, compression_ratio=0.8, max_tokens=2048):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.max_tokens = max_tokens
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, padded_bag, key_padding_mask, text_feature_batch):
        """
        SỬA LỖI: Nhận TÚI ĐÃ PAD (thay vì list)
        Args:
            padded_bag (Tensor): (B, N_patches, D)
            key_padding_mask (Tensor): (B, N_patches) - Mask (True là padding)
            text_feature_batch (Tensor): (B, D)
        """
        
        num_patches_per_bag = (~key_padding_mask).sum(dim=1) 
        k_per_bag = (num_patches_per_bag.float() * self.compression_ratio).int()
        k_per_bag = torch.clamp(k_per_bag, 1, self.max_tokens)
        k_per_bag = torch.min(k_per_bag, num_patches_per_bag)
        k_per_bag[k_per_bag == 0] = 1 # Đảm bảo k ít nhất là 1
        
        max_k = k_per_bag.max().item()
        if max_k == 0: max_k = 1 # Xử lý trường hợp tất cả túi đều rỗng
        
        text_q = self.q_proj(text_feature_batch).unsqueeze(1) 
        patches_k = self.k_proj(padded_bag) 
        
        scores = torch.bmm(text_q, patches_k.transpose(1, 2))
        scores = scores.squeeze(1) # (B, N)

        scores.masked_fill_(key_padding_mask, float('-inf'))
        
        _, top_k_indices = torch.topk(scores, k=max_k, dim=1)

        compressed_bag = torch.gather(
            padded_bag, 
            dim=1, 
            index=top_k_indices.unsqueeze(-1).expand(-1, -1, padded_bag.shape[-1])
        )
        
        new_key_padding_mask = torch.arange(max_k, device=padded_bag.device).expand(len(k_per_bag), max_k) >= k_per_bag.unsqueeze(1)

        return compressed_bag, new_key_padding_mask

class CrossModalAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True 
        )

    def forward(self, query_text, key_value_patches, key_padding_mask):
        attn_output, _ = self.attention(
            query=query_text,
            key=key_value_patches,
            value=key_value_patches,
            key_padding_mask=key_padding_mask
        )
        return attn_output