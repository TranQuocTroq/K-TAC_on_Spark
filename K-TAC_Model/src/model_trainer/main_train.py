# src/model_trainer/main_train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import warnings
from transformers import AutoTokenizer

# Import các file của bạn
from .utils import load_config, set_seed
from .dataset import WSIFocusDataset, custom_collate
from .architecture import FocusOnSpark
from .engine import train_one_epoch, evaluate

warnings.filterwarnings("ignore")

def get_split_indices(split_file_path, master_slide_ids, split_column):
    """
    Đọc file split (ví dụ: splits_0.csv) và lấy indices từ cột [split_column].
    """
    try:
        split_df = pd.read_csv(split_file_path)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file split: {split_file_path}")
        return []
        
    if split_column not in split_df.columns:
        print(f"LỖI CỘT: Không tìm thấy cột '{split_column}' trong file {split_file_path}")
        return []
        
    split_slide_ids = split_df[split_column].dropna().tolist() 
    indices = [master_slide_ids.index(sid) for sid in split_slide_ids if sid in master_slide_ids]
    return indices

def run_fold(fold_num, setting_name, dataset_name, config, master_dataset, tokenizer):
    """
    Hàm này chạy toàn bộ 1 fold (huấn luyện, đánh giá, và test).
    SỬA LỖI: Sẽ trả về 3 giá trị (AUC, F1, Accuracy)
    """
    print(f"\n--- Bắt đầu Fold {fold_num} cho {dataset_name} / {setting_name} ---")
    
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    
    # 1. XÂY DỰNG TÊN FOLDER PHẲNG
    split_folder_name = f"{dataset_name}_{setting_name}s_10folds"
    
    # 2. Lấy đường dẫn file split
    split_dir = os.path.join(
        config['processed_data_dir'], 
        config['split_base_dir'], 
        split_folder_name
    )
    split_file = os.path.abspath(os.path.join(split_dir, f"splits_{fold_num}.csv"))
    
    # 3. Lấy Indices cho Train, Val, VÀ TEST
    master_slide_ids = master_dataset.slide_ids
    train_indices = get_split_indices(split_file, master_slide_ids, "train")
    val_indices = get_split_indices(split_file, master_slide_ids, "val")
    test_indices = get_split_indices(split_file, master_slide_ids, "test")
    
    print(f"  Fold {fold_num}: Train {len(train_indices)}, Val {len(val_indices)}, Test {len(test_indices)} mẫu.")
    if not train_indices or not val_indices:
        print(f"  LỖI: Không có đủ mẫu train/val (File split {split_file} bị thiếu hoặc rỗng). Bỏ qua fold này.")
        # SỬA LỖI: Trả về 3 giá trị
        return 0.0, 0.0, 0.0 

    # 4. Tạo DataLoaders
    master_dataset.set_tokenizer(tokenizer) 
    train_subset = Subset(master_dataset, train_indices)
    val_subset = Subset(master_dataset, val_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'], 
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'], 
        collate_fn=custom_collate
    )
    
    test_loader = None
    if test_indices:
        test_subset = Subset(master_dataset, test_indices)
        test_loader = DataLoader(
            test_subset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'], 
            collate_fn=custom_collate
        )
    else:
        print("  CẢNH BÁO: Không tìm thấy 'test' split. Sẽ báo cáo kết quả trên 'val'.")
        test_loader = val_loader 
    
    # 5. Khởi tạo Mô hình (cho mỗi fold)
    local_config = config.copy()
    local_config['model']['num_classes'] = master_dataset.num_classes
    model = FocusOnSpark(local_config) # 1. Khởi tạo trên CPU

    # === SỬA LỖI: KIỂM TRA VÀ SỬ DỤNG NHIỀU GPU (T4x2) ===
    if torch.cuda.device_count() > 1:
        print(f"--- [Main] Phát hiện {torch.cuda.device_count()} GPUs! Đang sử dụng DataParallel. ---")
        model = torch.nn.DataParallel(model) # 2. Bọc model
    
    model = model.to(device) # 3. Đẩy model lên GPU

    # 6. Khởi tạo Optimizer và Loss
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # 7. Vòng lặp Huấn luyện
    best_val_auc = -1.0
    best_val_f1 = -1.0 
    best_val_acc = -1.0 # Thêm biến lưu Acc
    fold_run_name = f"{config['run_name']}_{dataset_name}_{setting_name}_fold_{fold_num}"
    save_path = f"checkpoints/{fold_run_name}_best.pt" 
    
    patience = config['training'].get('early_stopping_patience', 20) 
    counter = 0 
    
    for epoch in range(config['training']['epochs']): 
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, criterion, device, local_config['model']['num_classes'])
        
        print(f"  Epoch {epoch+1:02d}: Train Loss: {train_loss:.4f}, Val Loss: {metrics['loss']:.4f}, Val Acc: {metrics['accuracy']:.4f}, Val F1: {metrics['f1_score']:.4f}, Val AUC: {metrics['auc']:.4f}")

        # 8. Lưu checkpoint (model) tốt nhất dựa trên Val AUC
        if metrics['auc'] > best_val_auc:
            best_val_auc = metrics['auc']
            best_val_f1 = metrics['f1_score'] 
            best_val_acc = metrics['accuracy'] # Lưu cả Acc
            
            # === SỬA LỖI: LƯU STATE DICT CỦA .module KHI DÙNG DataParallel ===
            if isinstance(model, torch.nn.DataParallel):
                # Nếu dùng nhiều GPU, lưu state_dict của model bên trong
                torch.save(model.module.state_dict(), save_path)
            else:
                # Nếu chỉ dùng 1 GPU, lưu như bình thường
                torch.save(model.state_dict(), save_path)
                
            print(f"  *** Model tốt nhất (Val AUC: {best_val_auc:.4f}) đã lưu vào {save_path} ***")
            counter = 0
        else:
            counter += 1

        # 9. Kiểm tra điều kiện Early Stopping
        if counter >= patience:
            print(f"  --- Dừng sớm (EarlyStopping) tại Epoch {epoch+1} vì Val AUC không cải thiện sau {patience} epochs. ---")
            break 
            
    print(f"--- Huấn luyện Fold {fold_num} xong. Model tốt nhất đã lưu. ---")
    
    # === BƯỚC ĐÁNH GIÁ TRÊN TẬP TEST ===
    print(f"--- Đang tải model tốt nhất từ: {save_path} ---")
    try:
        # Tải checkpoint vào model GỐC (chưa bọc)
        final_model = FocusOnSpark(local_config).to(device)
        final_model.load_state_dict(torch.load(save_path))
        
        # BỌC LẠI model nếu cần (để chạy đánh giá test song song)
        if torch.cuda.device_count() > 1:
            print("--- [Test] Bọc model (DataParallel) để đánh giá trên Test set ---")
            final_model = torch.nn.DataParallel(final_model)
            
    except Exception as e:
        print(f"  LỖI: Không thể tải model đã lưu. Trả về kết quả val. Lỗi: {e}")
        return best_val_auc, best_val_f1, best_val_acc

    print("--- Bắt đầu đánh giá trên TẬP TEST ---")
    test_metrics = evaluate(final_model, test_loader, criterion, device, local_config['model']['num_classes'])
    
    print(f"  KẾT QUẢ TEST: Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['f1_score']:.4f}, Test AUC: {test_metrics['auc']:.4f}")
    
    # 10. Trả về KẾT QUẢ TEST (Cả 3 chỉ số)
    return test_metrics['auc'], test_metrics['f1_score'], test_metrics['accuracy'] 

def main():
    # 1. Tải Config
    config_path = "configs/model_config.yaml"
    config = load_config(config_path)
    if config is None:
        return
        
    # 2. Cài đặt Seed
    set_seed(config['seed'])
    
    # 3. Tạo thư mục
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 4. Tải Tokenizer (1 lần duy nhất)
    print("--- [Main] Tải Tokenizer (1 lần)... ---")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder_name'])
    print("--- [Main] Tải Tokenizer thành công ---")

    # 5. Vòng lặp THỰC NGHIỆM chính
    all_results = [] 
    
    for experiment in config['experiments_to_run']:
        dataset_name = experiment['dataset_name']
        num_classes = experiment['num_classes']
        
        # XÂY DỰNG ĐƯỜNG DẪN TUYỆT ĐỐI
        manifest_path_raw = os.path.join(config['manifest_base_dir'], dataset_name, experiment['manifest_file'])
        manifest_path = os.path.abspath(manifest_path_raw)
        
        prompts_path_raw = os.path.join(config['processed_data_dir'], config['prompt_base_dir'], experiment['prompts_file'])
        prompts_path = os.path.abspath(prompts_path_raw)
        
        print(f"\n========================================================")
        print(f"======== ĐANG TẢI DATASET: {dataset_name} ========")
        print(f"========================================================")
        
        # 6. Tải Master Dataset (cho bộ dữ liệu này)
        master_dataset = WSIFocusDataset(
            config=config,
            dataset_name=dataset_name,
            manifest_path=manifest_path, 
            prompts_path=prompts_path,   
            num_classes=num_classes
        )
        
        for setting_name in experiment['settings']: # Lặp qua "4shot", "8shot", ...
            print(f"\n****** Bắt đầu thực nghiệm: {dataset_name} / {setting_name} ******")
            
            fold_results_auc = []
            fold_results_f1 = []
            fold_results_acc = []
            
            for i in range(10): # Lặp qua 10 splits (0-9)
                test_auc, test_f1, test_acc = run_fold(
                    fold_num=i,
                    setting_name=setting_name,
                    dataset_name=dataset_name,
                    config=config,
                    master_dataset=master_dataset,
                    tokenizer=tokenizer
                )
                fold_results_auc.append(test_auc)
                fold_results_f1.append(test_f1)
                fold_results_acc.append(test_acc) # Lưu Acc
            
            # 7. Tính toán và lưu kết quả
            mean_auc = np.mean(fold_results_auc)
            std_auc = np.std(fold_results_auc)
            mean_f1 = np.mean(fold_results_f1)
            std_f1 = np.std(fold_results_f1)
            mean_acc = np.mean(fold_results_acc) # Tính Acc
            std_acc = np.std(fold_results_acc) # Tính Acc
            
            print(f"\n****** TỔNG KẾT (TỪ TẬP TEST): {dataset_name} / {setting_name} ******")
            print(f"  Các kết quả AUC (10-fold): {[round(f, 4) for f in fold_results_auc]}")
            print(f"  Acc Trung bình: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"  F1 Trung bình:  {mean_f1:.4f} ± {std_f1:.4f}")
            print(f"  AUC Trung bình: {mean_auc:.4f} ± {std_auc:.4f}")
            print("************************************************\n")
            
            all_results.append({
                "dataset": dataset_name,
                "setting": setting_name,
                "mean_acc": mean_acc, 
                "std_acc": std_acc,
                "mean_f1": mean_f1, 
                "std_f1": std_f1,
                "mean_auc": mean_auc,
                "std_auc": std_auc
            })

    # 8. In và lưu bảng kết quả cuối cùng
    log_df_path = f"logs/{config['run_name']}_summary.csv"
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(log_df_path, index=False)
    
    print("\n=================================================")
    print("======== TỔNG KẾT TOÀN BỘ THỰC NGHIỆM (TỪ TẬP TEST) ========")
    print(results_df.to_string()) 
    print(f"\n--- [Main] Kết thúc. Log tổng kết đã lưu vào {log_df_path} ---")

if __name__ == "__main__":
    main()