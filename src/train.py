import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import open_clip
import os

from utils import seed_everything, get_config
from dataset import download_and_extract_data, HFSlakeDataset
from models import MedVQABAN, MedVQABLIP2
from evaluate import evaluate_model

def train(args):
    seed_everything()
    cfg_main = get_config()
    cfg = cfg_main[args.model]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data Setup
    hf_ds = download_and_extract_data(cfg_main)
    _, _, transform = open_clip.create_model_and_transforms(cfg['clip_model'])
    
    # Tokenizer Selection
    if args.model == 'ban':
        tokenizer = AutoTokenizer.from_pretrained(cfg['bert_model'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg['llm_model'])
        tokenizer.pad_token = tokenizer.eos_token
        
    print(">>> Preparing Datasets...")
    train_ds = HFSlakeDataset(hf_ds['train'], cfg_main['data']['root_dir'], tokenizer, transform, is_train=True, max_seq_len=cfg['max_seq_len'])
    val_ds = HFSlakeDataset(hf_ds['validation'], cfg_main['data']['root_dir'], tokenizer, transform, label_encoder=train_ds.label_encoder, is_train=False, max_seq_len=cfg['max_seq_len'])
    
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)
    
    # Model Setup
    num_classes = len(train_ds.label_encoder.classes_)
    if args.model == 'ban':
        model = MedVQABAN(num_classes, cfg)
    else:
        model = MedVQABLIP2(num_classes, cfg)
    
    model = model.to(device)
    
    # Optimizer
    # We filter here to only update trainable params
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    print(f">>> Starting training for {args.model.upper()} on {device}...")
    
    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            imgs = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, input_ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{cfg['epochs']} - Loss: {avg_loss:.4f}")
        
        # Validation
        acc = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy: {acc:.2f}%")
        
    # --- FIXED SAVE LOGIC ---
    save_path = f"results/{args.model}_model.pth"
    print(f">>> Saving model to {save_path}...")
    
    # Only save parameters that require gradients (Trainable params)
    # This excludes the huge frozen LLM weights
    state_dict_to_save = {k: v for k, v in model.state_dict().items() if v.requires_grad}
    
    # Fallback: if state_dict has no requires_grad info (sometimes happens after processing),
    # we filter by matching keys with named_parameters
    if len(state_dict_to_save) == 0:
         trainable_keys = [n for n, p in model.named_parameters() if p.requires_grad]
         state_dict_to_save = {k: v for k, v in model.state_dict().items() if k in trainable_keys}

    torch.save(state_dict_to_save, save_path)
    print(">>> Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['ban', 'blip2'], help='Model architecture to train')
    args = parser.parse_args()
    os.makedirs('results', exist_ok=True)
    train(args)