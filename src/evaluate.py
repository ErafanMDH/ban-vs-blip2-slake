import torch
import numpy as np
from sklearn.metrics import accuracy_score

def evaluate_model(model, dataloader, device, verbose=False):
    model.eval()
    all_preds, all_labels = [], []
    all_types, all_questions = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(imgs, input_ids, mask)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if verbose:
                all_types.extend(batch['answer_type'])
                all_questions.extend(batch['question'])
                
    acc = accuracy_score(all_labels, all_preds) * 100
    
    if verbose:
        print("\n--- Detailed Evaluation ---")
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        all_types = np.array(all_types)
        
        # Type breakdown
        for q_type in ['CLOSED', 'OPEN']:
            mask = all_types == q_type
            if mask.sum() > 0:
                type_acc = accuracy_score(all_labels[mask], all_preds[mask]) * 100
                print(f"{q_type} Accuracy: {type_acc:.2f}% ({mask.sum()} samples)")
                
        # Spatial breakdown
        spatial_kws = ['where', 'location', 'side', 'left', 'right']
        spatial_mask = [any(k in q.lower() for k in spatial_kws) for q in all_questions]
        spatial_mask = np.array(spatial_mask)
        if spatial_mask.sum() > 0:
            sp_acc = accuracy_score(all_labels[spatial_mask], all_preds[spatial_mask]) * 100
            print(f"Spatial Accuracy: {sp_acc:.2f}% ({spatial_mask.sum()} samples)")
            
    return acc

if __name__ == "__main__":
    import argparse
    import os
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    import open_clip
    
    # Import from your other modules
    from utils import get_config, seed_everything
    from dataset import download_and_extract_data, HFSlakeDataset
    from models import MedVQABAN, MedVQABLIP2

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['ban', 'blip2'])
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    seed_everything()
    cfg_main = get_config()
    cfg = cfg_main[args.model]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Setup Data & Tokenizer
    hf_ds = download_and_extract_data(cfg_main)
    _, _, transform = open_clip.create_model_and_transforms(cfg['clip_model'])
    
    if args.model == 'ban':
        tokenizer = AutoTokenizer.from_pretrained(cfg['bert_model'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg['llm_model'])
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Re-create Validation Loader
    # We load train just to fit the label encoder so classes match the saved model
    print(">>> Setting up dataset...")
    train_ds_temp = HFSlakeDataset(hf_ds['train'], cfg_main['data']['root_dir'], tokenizer, transform, is_train=True)
    val_ds = HFSlakeDataset(hf_ds['validation'], cfg_main['data']['root_dir'], tokenizer, transform, label_encoder=train_ds_temp.label_encoder, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    # 3. Load Model Structure
    num_classes = len(train_ds_temp.label_encoder.classes_)
    if args.model == 'ban':
        model = MedVQABAN(num_classes, cfg)
    else:
        model = MedVQABLIP2(num_classes, cfg)

    # 4. Load Weights
    print(f">>> Loading weights from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # 5. Run Evaluation
    print(f">>> Running evaluation for {args.model}...")
    evaluate_model(model, val_loader, device, verbose=True)