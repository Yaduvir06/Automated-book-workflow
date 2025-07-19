import torch
from transformers import Phi3ForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader, Dataset
import json
import os
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

class PreferenceDataset(Dataset):
    """
    A dataset for preference data, where each sample consists of a prompt
    and two responses: one "good" (chosen) and one "bad" (rejected).
    """
    def __init__(self, data_file: str, tokenizer, max_len: int = 512):
        self.samples = [json.loads(line) for line in open(data_file, 'r', encoding='utf-8')]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Combine prompt and response for reward modeling
        chosen_text = sample["prompt"] + sample["good"]
        rejected_text = sample["prompt"] + sample["bad"]
        
        chosen_tokens = self.tokenizer(chosen_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        rejected_tokens = self.tokenizer(rejected_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        
        return {
            "input_ids_chosen": chosen_tokens["input_ids"].squeeze(0),
            "attention_mask_chosen": chosen_tokens["attention_mask"].squeeze(0),
            "input_ids_rejected": rejected_tokens["input_ids"].squeeze(0),
            "attention_mask_rejected": rejected_tokens["attention_mask"].squeeze(0),
        }

def train_reward_model(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    data_file: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "preferences.jsonl"),
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models", "reward_model"),
    epochs: int = 1,
    batch_size: int = 1,
    lr: float = 2e-5
):
    """
    Trains a reward model using preference data.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Load tokenizer and add a padding token if it doesn't exist.
    # This is crucial for sequence classification tasks.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # **FIX:** Use the specific `Phi3ForSequenceClassification` class instead of the auto class.
    # This avoids the ValueError because we are not relying on the auto-mapping.
    model = Phi3ForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=1,
        quantization_config=quantization_config,
        device_map="auto", 
        trust_remote_code=True
    )
    
    # Ensure the model's config also reflects the pad_token_id.
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare model for QLoRA training
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS # Important for sequence classification
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # No need for model.to(device) as device_map="auto" handles it.

    # Prepare dataset and dataloader
    dataset = PreferenceDataset(data_file, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            # Move batch to the same device as the model
            chosen_ids = batch["input_ids_chosen"].to(device)
            chosen_mask = batch["attention_mask_chosen"].to(device)
            rejected_ids = batch["input_ids_rejected"].to(device)
            rejected_mask = batch["attention_mask_rejected"].to(device)

            # Get scores from the model
            # The model outputs logits for the chosen and rejected sequences
            scores_chosen = model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
            scores_rejected = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits
            
            # Compute the ranking loss
            # We want the score of the chosen sequence to be higher than the rejected one
            loss = -torch.nn.functional.logsigmoid(scores_chosen - scores_rejected).mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{len(loader)}, Loss: {loss.item():.4f}")

    print("✅ Finished training reward model.")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    train_reward_model()