
# src/model_trainer/architecture.py

import torch
import torch.nn as nn
from transformers import AutoModel
from .modules import KAVTC_Module_G2, CrossModalAggregator 

class FocusOnSpark(nn.Module):
    """
    SỬA LỖI ĐA GPU: Sửa logic 'forward' để nhận TENSOR đã pad.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config['model'] 

        self.text_encoder = AutoModel.from_pretrained(
            self.config['text_encoder_name']
        )

        text_embed_dim = self.text_encoder.config.hidden_size

        self.image_feature_dim = self.config['image_feature_dim']
        self.projection_dim = self.config['projection_dim']

        self.image_proj = nn.Linear(self.image_feature_dim, self.projection_dim)
        self.text_proj = nn.Linear(text_embed_dim, self.projection_dim)
        self.norm = nn.LayerNorm(self.projection_dim)
        self.relu = nn.ReLU()

        self.kavtc = KAVTC_Module_G2(
            embed_dim=self.projection_dim,
            compression_ratio=self.config['kavtc']['compression_ratio'],
            max_tokens=self.config['kavtc']['max_tokens']
        )

        self.aggregator = CrossModalAggregator(
            embed_dim=self.projection_dim,
            num_heads=self.config['aggregator']['num_heads'],
            dropout=self.config['aggregator']['dropout']
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(self.projection_dim, self.projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config['head_dropout']),
            nn.Linear(self.projection_dim // 2, self.config['num_classes'])
        )

    def forward(self, padded_bags, key_padding_mask, input_ids, attention_mask):
        """
        SỬA LỖI ĐA GPU:
        Nhận 'padded_bags' và 'key_padding_mask' trực tiếp từ collate_fn.
        """
        # 1. Mã hóa Text
        text_output = self.text_encoder(input_ids, attention_mask)
        text_feature = text_output.pooler_output # (B, 768)

        # 2. Chiếu (Project) về không gian chung
        text_feature_proj = self.relu(self.norm(self.text_proj(text_feature))) # (B, 256)

        # Chiếu túi đã pad
        bag_features_proj = self.relu(self.norm(self.image_proj(padded_bags))) # (B, N, 256)

        # 3. Áp dụng KAVTC (Giai đoạn 2)
        compressed_bags, new_key_padding_mask = self.kavtc(
            bag_features_proj, 
            key_padding_mask, 
            text_feature_proj
        )

        # 4. Tổng hợp (Aggregation)
        query = text_feature_proj.unsqueeze(1) # (B, 1, 256)

        aggregated_vector = self.aggregator(
            query, 
            compressed_bags, 
            new_key_padding_mask 
        )

        aggregated_vector = aggregated_vector.squeeze(1) # (B, 256)

        # 5. Dự đoán
        logits = self.prediction_head(aggregated_vector) # (B, num_classes)

        return logits
