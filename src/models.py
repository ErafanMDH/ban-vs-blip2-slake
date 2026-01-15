import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import open_clip
from transformers import AutoModel, AutoModelForCausalLM, Blip2QFormerModel, Blip2QFormerConfig
from peft import get_peft_model, LoraConfig, TaskType

# --- Helper Function for Unfreezing ---
def unfreeze_last_visual_block(clip_model):
    """
    Smartly finds the last visual block to unfreeze, handling 
    BiomedCLIP (TimmModel) vs Standard CLIP structures.
    """
    visual = clip_model.visual
    target_blocks = None
    
    # 1. Check for BiomedCLIP / TimmModel Wrapper
    if hasattr(visual, 'trunk') and hasattr(visual.trunk, 'blocks'):
        target_blocks = visual.trunk.blocks
        print(">>> Detected BiomedCLIP (TimmModel). Unfreezing last block of 'trunk.blocks'.")
        
    # 2. Check for Standard OpenCLIP ViT
    elif hasattr(visual, 'blocks'):
        target_blocks = visual.blocks
        print(">>> Detected Standard ViT. Unfreezing last block of 'blocks'.")
        
    # 3. Check for OpenAI CLIP
    elif hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks'):
        target_blocks = visual.transformer.resblocks
        print(">>> Detected OpenAI CLIP. Unfreezing last block of 'transformer.resblocks'.")
        
    # Apply Unfreeze
    if target_blocks is not None:
        for p in target_blocks[-1:].parameters():
            p.requires_grad = True
    else:
        print(">>> WARNING: Could not find visual blocks. Vision encoder remains frozen.")

# --- BAN Components ---
class BCNet(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, k_glimpses=2):
        super(BCNet, self).__init__()
        self.c = h_dim
        self.k = k_glimpses
        self.v_net = weight_norm(nn.Linear(v_dim, h_dim * self.k), dim=None)
        self.q_net = weight_norm(nn.Linear(q_dim, h_dim * self.k), dim=None)
        self.p_net = weight_norm(nn.Linear(h_dim, h_out), dim=None)

    def forward(self, v, q):
        v_proj = self.v_net(v)
        q_proj = self.q_net(q)
        logits = torch.matmul(v_proj, q_proj.transpose(1, 2))
        v_ = v_proj.view(v.size(0), -1, self.k, self.c)
        q_ = q_proj.view(q.size(0), -1, self.k, self.c)
        att_maps = F.softmax(torch.einsum('bnkd,bmkd->bnmk', v_, q_), dim=2)
        q_weighted = torch.einsum('bnmk,bmkd->bnkd', att_maps, q_)
        f = (v_ * q_weighted).sum(dim=1)
        return self.p_net(f).view(v.size(0), -1)

class MedVQABAN(nn.Module):
    def __init__(self, num_classes, cfg):
        super(MedVQABAN, self).__init__()
        print(">>> Loading BiomedCLIP & Bio_ClinicalBERT...")
        self.clip, _, _ = open_clip.create_model_and_transforms(cfg['clip_model'])
        self.text_model = AutoModel.from_pretrained(cfg['bert_model'])
        
        # 1. Freeze All First
        for p in self.clip.parameters(): p.requires_grad = False
        
        # 2. Smart Unfreeze
        unfreeze_last_visual_block(self.clip)
        
        self.img_proj = nn.Linear(512, cfg['hidden_dim'])
        self.ban = BCNet(cfg['hidden_dim'], 768, cfg['hidden_dim'], cfg['hidden_dim'], cfg['glimpses'])
        self.classifier = nn.Sequential(
            nn.Linear(cfg['glimpses'] * cfg['hidden_dim'], cfg['hidden_dim']),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(cfg['hidden_dim'], num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # Allow gradients for the unfrozen part
        img_feats = self.clip.encode_image(images).unsqueeze(1) 
        img_feats = self.img_proj(img_feats)
        text_feats = self.text_model(input_ids, attention_mask).last_hidden_state
        joint_feat = self.ban(img_feats, text_feats)
        return self.classifier(joint_feat)

# --- BLIP-2 with LoRA ---
class MedVQABLIP2(nn.Module):
    def __init__(self, num_classes, cfg):
        super(MedVQABLIP2, self).__init__()
        print(f">>> Loading BiomedCLIP, Q-Former & {cfg['llm_model']}...")
        
        # 1. Vision Encoder
        self.clip, _, _ = open_clip.create_model_and_transforms(cfg['clip_model'])
        
        # Freeze & Smart Unfreeze
        for p in self.clip.parameters(): p.requires_grad = False
        unfreeze_last_visual_block(self.clip)
        
        # 2. Q-Former
        self.qformer = Blip2QFormerModel(Blip2QFormerConfig(
            hidden_size=cfg['hidden_dim'], encoder_hidden_size=512, 
            num_hidden_layers=6, num_attention_heads=12
        ))
        self.query_tokens = nn.Parameter(torch.zeros(1, cfg['num_query_tokens'], cfg['hidden_dim']))
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # 3. LLM (BioGPT or Opt)
        self.llm = AutoModelForCausalLM.from_pretrained(cfg['llm_model'])
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=cfg['lora_r'], 
            lora_alpha=cfg['lora_alpha'], 
            lora_dropout=cfg['lora_dropout'],
            # --- FIX: Explicitly target BioGPT's attention layers ---
            target_modules=["q_proj", "v_proj"]
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()
        
        llm_hidden_size = self.llm.config.hidden_size
        self.llm_proj = nn.Linear(cfg['hidden_dim'], llm_hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(llm_hidden_size, cfg['hidden_dim']),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(cfg['hidden_dim'], num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        B = images.size(0)
        
        # 1. Vision Features (Gradients allowed)
        vision_features = self.clip.encode_image(images).unsqueeze(1)
            
        # 2. Q-Former
        query_tokens = self.query_tokens.expand(B, -1, -1)
        visual_tokens = self.qformer(
            query_embeds=query_tokens, encoder_hidden_states=vision_features
        ).last_hidden_state
        
        visual_proj = self.llm_proj(visual_tokens)
        
        # 4. LLM Embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_proj, inputs_embeds], dim=1)
        
        vis_mask = torch.ones(B, visual_proj.size(1), device=images.device)
        full_mask = torch.cat([vis_mask, attention_mask], dim=1)
        
        llm_out = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_mask, output_hidden_states=True)
        
        text_lengths = attention_mask.sum(dim=1) 
        last_token_indices = visual_proj.size(1) + text_lengths - 1
        
        hidden_states = llm_out.hidden_states[-1]
        pooled = hidden_states[torch.arange(B), last_token_indices, :]
        
        return self.classifier(pooled)