import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_config():
    return {
        'ban': {
            'batch_size': 64,  # Lower slightly if unfreezing vision
            'learning_rate': 2e-5,
            'epochs': 20,      # 30 might be overfitting
            'max_seq_len': 32,
            'hidden_dim': 768,
            'glimpses': 2,
            'clip_model': 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            'bert_model': 'emilyalsentzer/Bio_ClinicalBERT'
        },
        'blip2': {
            'batch_size': 16,
            'learning_rate': 2e-5, # Slightly lower LR for BioGPT
            'epochs': 20,
            'max_seq_len': 128,    # Generative models need more room
            'hidden_dim': 768,
            'num_query_tokens': 32,
            'clip_model': 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            
            # --- MAJOR CHANGE: Use a Medical LLM ---
            'llm_model': 'microsoft/BioGPT', 
            
            # --- LoRA Config ---
            'lora_r': 32,          
            'lora_alpha': 64,      
            'lora_dropout': 0.05   
        },
        'data': {
            'root_dir': './data/imgs',
            'extract_path': './data'
        }
    }