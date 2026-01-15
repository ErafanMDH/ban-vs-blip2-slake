import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import open_clip
import pandas as pd
import numpy as np
import os
import argparse
import gc
import warnings
from collections import Counter

from utils import seed_everything, get_config
from dataset import download_and_extract_data, HFSlakeDataset
from models import MedVQABAN, MedVQABLIP2

warnings.filterwarnings("ignore")

# --- Helper: Simple BLEU-1 Calculator ---
def calculate_bleu1(reference, candidate):
    """Computes BLEU-1 score: Precision of unigrams."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    common = Counter(ref_tokens) & Counter(cand_tokens)
    num_same = sum(common.values())
    
    return num_same / len(cand_tokens)

def get_pathology_mask(questions):
    """Filter for pathology/disease related questions."""
    keywords = [
        'abnormal', 'abnormality', 'pathology', 'disease', 'finding', 
        'wrong', 'diagnosis', 'pneumonia', 'edema', 'fracture', 'condition',
        'illness', 'cancer', 'tumor', 'lesion', 'infection'
    ]
    return [any(k in q.lower() for k in keywords) for q in questions]

def get_spatial_mask(questions):
    """Filter for spatial/location related questions."""
    keywords = ['where', 'location', 'side', 'left', 'right', 'position', 'plane']
    return [any(k in q.lower() for k in keywords) for q in questions]

def run_inference(model_name, checkpoint_path, cfg_main, label_encoder, device):
    print(f"\n>>> Processing {model_name.upper()}...")
    cfg = cfg_main[model_name]
    
    # 1. Setup Tokenizer & Transform
    _, _, transform = open_clip.create_model_and_transforms(cfg['clip_model'])
    
    if model_name == 'ban':
        tokenizer = AutoTokenizer.from_pretrained(cfg['bert_model'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg['llm_model'])
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Setup Dataset
    hf_ds = download_and_extract_data(cfg_main)
    val_ds = HFSlakeDataset(
        hf_ds['validation'], 
        cfg_main['data']['root_dir'], 
        tokenizer, 
        transform, 
        label_encoder=label_encoder, 
        is_train=False, 
        max_seq_len=cfg['max_seq_len']
    )
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

    # 3. Load Model
    num_classes = len(label_encoder.classes_)
    if model_name == 'ban':
        model = MedVQABAN(num_classes, cfg)
    else:
        model = MedVQABLIP2(num_classes, cfg)
        
    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 4. Inference
    predictions_text = []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            outputs = model(imgs, input_ids, mask)
            _, predicted_ids = torch.max(outputs, 1)
            preds = label_encoder.inverse_transform(predicted_ids.cpu().numpy())
            predictions_text.extend(preds)

    # Cleanup
    del model, val_loader, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return predictions_text

def compute_metrics(df, pred_col, gt_col, q_col, type_col):
    """Calculates the 5 requested metrics."""
    # 1. Overall Accuracy
    acc = (df[pred_col] == df[gt_col]).mean() * 100
    
    # 2. Closed Accuracy
    closed_mask = df[type_col] == 'CLOSED'
    acc_closed = (df[pred_col][closed_mask] == df[gt_col][closed_mask]).mean() * 100 if closed_mask.sum() > 0 else 0
    
    # 3. Open BLEU-1
    open_mask = df[type_col] == 'OPEN'
    if open_mask.sum() > 0:
        bleu_scores = [
            calculate_bleu1(ref, cand) 
            for ref, cand in zip(df[gt_col][open_mask], df[pred_col][open_mask])
        ]
        bleu_open = np.mean(bleu_scores) * 100
    else:
        bleu_open = 0
        
    # 4. Pathology Accuracy
    path_mask = np.array(get_pathology_mask(df[q_col]))
    acc_path = (df[pred_col][path_mask] == df[gt_col][path_mask]).mean() * 100 if path_mask.sum() > 0 else 0
    
    # 5. Spatial Accuracy
    spat_mask = np.array(get_spatial_mask(df[q_col]))
    acc_spat = (df[pred_col][spat_mask] == df[gt_col][spat_mask]).mean() * 100 if spat_mask.sum() > 0 else 0
    
    return acc, acc_closed, bleu_open, acc_path, acc_spat

def main(args):
    seed_everything()
    cfg_main = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(">>> Initializing Label Encoder...")
    hf_ds = download_and_extract_data(cfg_main)
    dummy_tokenizer = AutoTokenizer.from_pretrained(cfg_main['ban']['bert_model'])
    _, _, dummy_transform = open_clip.create_model_and_transforms(cfg_main['ban']['clip_model'])
    
    train_ds = HFSlakeDataset(hf_ds['train'], cfg_main['data']['root_dir'], dummy_tokenizer, dummy_transform, is_train=True)
    label_encoder = train_ds.label_encoder
    
    # Get Metadata
    print(">>> Extracting Metadata...")
    val_raw_data = [item for item in hf_ds['validation'] if item['q_lang'] == 'en']
    questions = [item['question'] for item in val_raw_data]
    ground_truth = [item['answer'] for item in val_raw_data]
    img_names = [item['img_name'] for item in val_raw_data]
    q_types = [item['answer_type'] for item in val_raw_data]

    # Run Models
    ban_preds = run_inference('ban', args.ban_checkpoint, cfg_main, label_encoder, device)
    blip_preds = run_inference('blip2', args.blip_checkpoint, cfg_main, label_encoder, device)

    # Create DataFrame
    df = pd.DataFrame({
        'Image_ID': img_names,
        'Question_Type': q_types,
        'Question': questions,
        'Ground_Truth': ground_truth,
        'BAN_Prediction': ban_preds,
        'BLIP2_Prediction': blip_preds
    })
    
    # Add correctness for filtering
    df['BAN_Correct'] = df['BAN_Prediction'] == df['Ground_Truth']
    df['BLIP2_Correct'] = df['BLIP2_Prediction'] == df['Ground_Truth']
    
    # --- METRIC REPORT ---
    ban_metrics = compute_metrics(df, 'BAN_Prediction', 'Ground_Truth', 'Question', 'Question_Type')
    blip_metrics = compute_metrics(df, 'BLIP2_Prediction', 'Ground_Truth', 'Question', 'Question_Type')
    
    print("\n" + "="*80)
    print(f"{'EVALUATION REPORT':^80}")
    print("="*80)
    print(f"{'Metric':<20} | {'BAN Model':<15} | {'BLIP-2 (BioGPT)':<15}")
    print("-" * 60)
    print(f"{'1. Overall Acc':<20} | {ban_metrics[0]:6.2f}%         | {blip_metrics[0]:6.2f}%")
    print(f"{'2. Closed Acc':<20} | {ban_metrics[1]:6.2f}%         | {blip_metrics[1]:6.2f}%")
    print(f"{'3. Open (BLEU-1)':<20} | {ban_metrics[2]:6.2f}          | {blip_metrics[2]:6.2f}")
    print(f"{'4. Pathology Acc':<20} | {ban_metrics[3]:6.2f}%         | {blip_metrics[3]:6.2f}%")
    print(f"{'5. Spatial Acc':<20} | {ban_metrics[4]:6.2f}%         | {blip_metrics[4]:6.2f}%")
    print("="*80)

    # --- SAMPLE DISAGREEMENTS ---
    print("\n" + "="*80)
    print(">>> 1. Disagreements (General) - Top 5 Samples")
    print("="*80)
    cols_to_show = ['Image_ID', 'Question', 'Ground_Truth', 'BAN_Prediction', 'BLIP2_Prediction']
    
    disagreements = df[df['BAN_Prediction'] != df['BLIP2_Prediction']].head(5)
    if not disagreements.empty:
        print(disagreements[cols_to_show].to_string(index=False))
    else:
        print("No disagreements found.")

    # --- SAMPLE OPEN-ENDED ERRORS ---
    print("\n" + "="*80)
    print(">>> 2. Open-Ended Errors (Where at least one model is WRONG) - 5 Samples")
    print("="*80)
    
    # Filter: Type is OPEN AND (BAN is Wrong OR BLIP2 is Wrong)
    error_mask = (df['Question_Type'] == 'OPEN') & ((df['BAN_Correct'] == False) | (df['BLIP2_Correct'] == False))
    open_errors = df[error_mask]
    
    if not open_errors.empty:
        # Pick 5 random samples if available, otherwise all of them
        n_samples = min(5, len(open_errors))
        print(open_errors[cols_to_show].sample(n=n_samples, random_state=42).to_string(index=False))
    else:
        print("Amazing! No errors found in Open-Ended questions.")

    # Save CSV
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/model_comparison.csv', index=False)
    print(f"\n[INFO] Detailed CSV saved to results/model_comparison.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ban_checkpoint', default='results/ban_model.pth')
    parser.add_argument('--blip_checkpoint', default='results/blip2_model.pth')
    args = parser.parse_args()
    main(args)