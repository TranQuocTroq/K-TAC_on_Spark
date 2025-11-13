# src/model_trainer/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import h5py
import os
import yaml
from transformers import AutoTokenizer

class WSIFocusDataset(Dataset):
    def __init__(self, config, dataset_name, manifest_path, prompts_path, num_classes):
        self.config = config
        self.num_classes = num_classes
        self.data_dir = os.path.join(
            config['processed_data_dir'],
            config['dataset_base_dir'],
            dataset_name
        ) 
        try:
            self.master_labels = pd.read_csv(manifest_path, encoding="utf-8")
            self.master_labels.set_index('slide_id', inplace=True)
            self.slide_ids = self.master_labels.index.tolist()
        except Exception as e:
            print(f"LỖI: Không tìm thấy file Master Labels: {manifest_path}. Lỗi: {e}")
            raise
        try:
            self.prompts_df = pd.read_csv(prompts_path, header=None, encoding="utf-8")
            self.text_prompts_list = self.prompts_df.iloc[:, 0].tolist()
        except Exception as e:
            print(f"LỖI: Không tìm thấy file Text Prompts: {prompts_path}. Lỗi: {e}")
            raise
        self.tokenizer = None
        self.max_text_len = self.config['model']['max_text_length']
        self.image_feature_dim = self.config['model']['image_feature_dim']

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        try:
            row = self.master_labels.loc[slide_id]
        except KeyError:
            print(f"LỖI: Không tìm thấy slide_id '{slide_id}' trong file labels.csv")
            return None
        label_id = row['label_id']
        label_tensor = torch.tensor(label_id, dtype=torch.long)
        try:
            high_res_prompt_index = label_id + self.num_classes
            high_res_prompt = self.text_prompts_list[high_res_prompt_index]
        except IndexError:
            print(f"LỖI: Không thể lấy prompt cho label_id {label_id}. (Index {high_res_prompt_index})")
            high_res_prompt = "" 
        if self.tokenizer is None:
            return None
        tokenized_text = self.tokenizer(
            high_res_prompt,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt"
        )
        input_ids = tokenized_text['input_ids'].squeeze(0)
        attention_mask = tokenized_text['attention_mask'].squeeze(0)
        h5_filename = f"{slide_id}.h5" 
        h5_path = os.path.join(self.data_dir, "h5", h5_filename) 
        try:
            with h5py.File(h5_path, 'r') as hf:
                bag_features = hf['features'][:] 
            if bag_features.shape[0] == 0:
                 bag_features_tensor = torch.empty((0, self.image_feature_dim), dtype=torch.float32)
            else:
                 bag_features_tensor = torch.tensor(bag_features, dtype=torch.float32)
        except Exception as e:
            print(f"LỖI H5: Không đọc được tệp H5: {h5_path}. Lỗi: {e}")
            return None 
        return {
            "bag_features": bag_features_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label_tensor,
            "wsi_id": slide_id
        }
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

def custom_collate(batch):
    """
    SỬA LỖI ĐA GPU:
    Hàm này sẽ pad (đệm) các túi (bags) để tạo ra 1 TENSOR duy nhất.
    DataParallel có thể "chia" (scatter) Tensors, nhưng không thể chia 'list'.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    labels = torch.stack([item['label'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    wsi_ids = [item['wsi_id'] for item in batch]
    bag_features_list = [item['bag_features'] for item in batch]
    non_empty_bags = [bag for bag in bag_features_list if bag.shape[0] > 0]
    if not non_empty_bags:
        feature_dim = batch[0]['bag_features'].shape[-1] if batch[0]['bag_features'].shape else 2048 # Lấy dim (2048)
        padded_bags = torch.empty((len(batch), 0, feature_dim), dtype=torch.float32)
        key_padding_mask = torch.ones((len(batch), 0), dtype=torch.bool)
    else:
        max_len = max(bag.shape[0] for bag in non_empty_bags)
        feature_dim = non_empty_bags[0].shape[1]
        padded_bags = torch.zeros(len(batch), max_len, feature_dim, dtype=torch.float32)
        key_padding_mask = torch.ones(len(batch), max_len, dtype=torch.bool)
        for i, bag in enumerate(bag_features_list):
            if bag.shape[0] > 0:
                padded_bags[i, :bag.shape[0], :] = bag
                key_padding_mask[i, :bag.shape[0]] = False # False = Data, True = Padding
    
    return {
        "padded_bags": padded_bags,
        "key_padding_mask": key_padding_mask,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "label": labels,
        "wsi_id": wsi_ids
    }