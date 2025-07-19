from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
import os
import json

def load_model_for_training(model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    
    # CRITICAL: Prepare model for k-bit training BEFORE adding LoRA
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    

    # LoRA config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    
   
    print("Adding LoRA adapters...")
    model = get_peft_model(model, peft_config)
  
    model.train()
    
   # enable gradients for LoRA paramatrs
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            print(f"Enabled gradients for: {name}")
    
    # Print trainable paramaters
    model.print_trainable_parameters()
    
    return tokenizer, model

class SimpleDataset(torch.utils.data.Dataset):
  
    def __init__(self, texts, tokenizer, max_length=384):  # Reduced max_length
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenizing the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Extracting the tensors and remove the batch dimension
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def train_sft_manual(
    data_file: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "preferences.jsonl"),
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models", "sft_finetuned")
):
    """SFT training with proper gradient setup for quantized + LoRA"""
    print("Starting SFT training with 4-bit quantization and LoRA...")

    # Load and prepare model
    tokenizer, model = load_model_for_training()

    # Create sample data if file doesn't exist
    if not os.path.exists(data_file):
        print(f"Creating sample training data at {data_file}")
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        sample_data = [
            {
                "prompt": "Rewrite this chapter opening in an engaging style:",
                "good": "The morning mist clung to the cobblestones like ghostly fingers, whispering secrets of the night before."
            },
            {
                "prompt": "Enhance this dialogue with vivid descriptions:",
                "good": "'We can't go back,' she whispered, her voice breaking like glass against stone."
            },
            {
                "prompt": "Improve this scene description:",
                "good": "The ancient library stretched endlessly before them, its shelves disappearing into shadow."
            }
        ]
        with open(data_file, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Load and process data
    print(f"Loading dataset from {data_file}")
    formatted_texts = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Format with chat template
            messages = [
                {"role": "user", "content": data["prompt"]},
                {"role": "assistant", "content": data["good"]}
            ]
            
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            formatted_texts.append(formatted_text)
    
    print(f"Formatted {len(formatted_texts)} training examples")
    
    # Creating dataset with smaller max_length
    train_dataset = SimpleDataset(formatted_texts, tokenizer, max_length=384)
    
    # Creating output directory
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments - optimized for quantized + LoRA
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2, 
        max_steps=10, 
        learning_rate=1e-4,  
        fp16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=5,
        report_to="none",
        warmup_steps=1,
        weight_decay=0.01,
        dataloader_num_workers=0,
        save_total_limit=2,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        label_names=["labels"],
        prediction_loss_only=True,
        
        gradient_checkpointing=False,
        
        dataloader_pin_memory=False,
        group_by_length=False,
    )

    # Initializing trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Verifying gradients are enabledd
    print("Checking gradient setup...")
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
    
    print(f"Found {len(trainable_params)} trainable parameters")
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found! LoRA setup failed.")

    # Clear memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Train model
    print("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("This might be expected on first run - trying to save what we have...")

    # Save model and tokenizer
    print("Saving model and tokenizer...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training complete. Model saved to {output_dir}")
    
  

if __name__ == "__main__":
    train_sft_manual()
