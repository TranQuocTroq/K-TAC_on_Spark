# src/model_trainer/utils.py

import torch
import random
import os
import numpy as np
import yaml

def set_seed(seed=42):
    """
    Cài đặt seed cho mọi thư viện để đảm bảo kết quả có thể tái lập.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # nếu dùng nhiều GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"--- [Utils] Đã cài đặt Seed: {seed} ---")

def load_config(config_path):
    """
    Tải file cấu hình .yaml
    Sử dụng encoding="utf-8" để giải quyết lỗi Unicode.
    """
    print(f"--- [Utils] Đang tải Config từ: {config_path} ---")
    
    # Mở file với encoding="utf-8"
    with open(config_path, 'r', encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as exc:
            print(f"LỖI: Lỗi cú pháp trong file config YAML: {exc}")
            return None

def save_checkpoint(epoch, model, optimizer, loss, checkpoint_path):
    """
    Lưu checkpoint của mô hình.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, checkpoint_path)
    # ĐÃ DỌN DẸP LOG: Chỉ in 1 dòng khi lưu xong
    # print(f"--- [Utils] Đã lưu checkpoint: {checkpoint_path} ---")