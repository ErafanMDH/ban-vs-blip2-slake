import os
import zipfile
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import LabelEncoder

def download_and_extract_data(config):
    """Downloads SLAKE dataset images if not present."""
    if not os.path.exists(config['data']['root_dir']):
        print(">>> Downloading images...")
        zip_path = hf_hub_download(repo_id="BoKelvin/SLAKE", filename="imgs.zip", repo_type="dataset")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(config['data']['extract_path'])
        print(">>> Extraction complete.")
    else:
        print(">>> Images already found.")

    return load_dataset("BoKelvin/SLAKE")

class HFSlakeDataset(Dataset):
    def __init__(self, hf_dataset, img_root_dir, tokenizer, transform=None, label_encoder=None, is_train=True, max_seq_len=32):
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Filter for English questions
        self.data = [item for item in hf_dataset if item['q_lang'] == 'en']

        if is_train and label_encoder is None:
            all_answers = [item['answer'] for item in self.data]
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(all_answers)
        else:
            self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_path = os.path.join(self.img_root_dir, item['img_name'])

        try:
            image = Image.open(full_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        inputs = self.tokenizer(
            item['question'],
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        try:
            label = self.label_encoder.transform([item['answer']])[0]
        except ValueError:
            label = 0 # Handle unseen labels in validation

        return {
            'image': image,
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long),
            'answer_type': item['answer_type'], # For evaluation
            'base_type': item['base_type'],     # For evaluation
            'question': item['question']        # For evaluation
        }