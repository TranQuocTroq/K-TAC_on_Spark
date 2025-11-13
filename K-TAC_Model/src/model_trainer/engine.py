# src/model_trainer/engine.py

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    SỬA LỖI ĐA GPU: Sửa cách truyền 'batch' vào model.
    """
    model.train() 
    total_loss = 0
    all_labels = []
    all_preds = []
    
    pbar = tqdm(dataloader, desc="Training Epoch", leave=False)
    for batch in pbar:
        if batch is None:
            continue 
            
        # 1. Chuyển TẤT CẢ Tensors trong batch sang GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        padded_bags = batch['padded_bags'].to(device)
        key_padding_mask = batch['key_padding_mask'].to(device)

        optimizer.zero_grad()
        
        # 2. SỬA LỖI: Truyền các Tensors đã pad (thay vì list)
        logits = model(padded_bags, key_padding_mask, input_ids, attention_mask)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
        pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})
        
    avg_loss = total_loss / (len(pbar) + 1e-6)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc

def evaluate(model, dataloader, criterion, device, num_classes):
    """
    SỬA LỖI ĐA GPU: Sửa cách truyền 'batch' vào model.
    """
    model.eval() 
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad(): 
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            if batch is None:
                continue
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            padded_bags = batch['padded_bags'].to(device)
            key_padding_mask = batch['key_padding_mask'].to(device)

            # 2. SỬA LỖI: Truyền các Tensors đã pad
            logits = model(padded_bags, key_padding_mask, input_ids, attention_mask)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / (len(pbar) + 1e-6)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) 
    
    auc = 0.0
    try:
        if num_classes == 2:
            probs_for_auc = np.array(all_probs)[:, 1] 
            auc = roc_auc_score(all_labels, probs_for_auc)
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError as e:
        auc = 0.0 
    
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "f1_score": f1,
        "auc": auc,
        "confusion_matrix": cm
    }
    
    return metrics