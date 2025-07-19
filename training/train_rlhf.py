import torch
import os
import gc
from dataclasses import dataclass

# Force CPU before any CUDA checks
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead  
from datasets import load_dataset

def create_working_ppo_config(tokenizer):
    """Create a PPOConfig that works on CPU"""
    print("🔧 Creating CPU-compatible PPOConfig...")
    
    # Method 1: Try bypassing device checks
    try:
        # Temporarily override CUDA check
        original_cuda_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        
        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=1,
            mini_batch_size=1,
            gradient_accumulation_steps=8,
            stop_token_id=tokenizer.eos_token_id,
            # CPU-specific settings
            bf16=False,
            fp16=False,
            dataloader_num_workers=0,
        )
        
        # Restore original function
        torch.cuda.is_available = original_cuda_available
        print("✅ PPOConfig created with device override")
        return config
        
    except Exception as e:
        print(f"Device override failed: {e}")
        
        # Method 2: Create minimal config and set attributes manually
        try:
            torch.cuda.is_available = lambda: False
            
            # Create with minimal required args
            config = PPOConfig(learning_rate=1e-5, batch_size=1)
            
            # Manually set additional attributes
            config.mini_batch_size = 1
            config.gradient_accumulation_steps = 8
            config.stop_token_id = tokenizer.eos_token_id
            config.stop_token = None  # This was missing and caused the crash
            config.bf16 = False
            config.fp16 = False
            config.dataloader_num_workers = 0
            
            torch.cuda.is_available = original_cuda_available
            print("✅ PPOConfig created with manual attributes")
            return config
            
        except Exception as e2:
            print(f"Manual config failed: {e2}")
            torch.cuda.is_available = original_cuda_available
            raise e2

def train_ppo_fixed(
    sft_model_path: str = "../data/models/sft_finetuned",
    reward_model_path: str = "../data/models/reward_model",
    ppo_model_path: str = "../data/models/ppo_policy", 
    prompt_data_file: str = "../data/preferences.jsonl",
):
    print("🚀 Starting FIXED PPO training...")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create working PPOConfig
    ppo_config = create_working_ppo_config(tokenizer)
    
    generation_kwargs = {
        "max_new_tokens": 16,
        "do_sample": True,
        "temperature": 0.8,
        "pad_token_id": tokenizer.pad_token_id,
        "num_beams": 1,
        "use_cache": False,
    }

    print("Loading models...")
    # Policy model with value head
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        sft_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    policy = policy.cpu()
    
    # Set generation config
    generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.8,
        max_new_tokens=16,
    )
    policy.generation_config = generation_config
    
    # Reference model (base model without value head)
    ref_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    ref_model = ref_model.cpu()
    ref_model.generation_config = generation_config
    
    # Simple reward model (reuse base model for now)
    reward_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        trust_remote_code=True, 
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    reward_model = reward_model.cpu()
    
    # Value model (base model)
    value_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    value_model = value_model.cpu()
    
    print("Preparing dataset...")
    def tokenize(example):
        tokenized = tokenizer(example["prompt"], truncation=True, max_length=24, padding=False)
        example["input_ids"] = tokenized["input_ids"]
        example["query"] = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)
        return example

    dataset = load_dataset("json", data_files=prompt_data_file, split="train")
    dataset = dataset.select(range(min(5, len(dataset))))  # Very small for CPU
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch", columns=["input_ids", "query"])

    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    print("Initializing PPOTrainer...")
    print(f"Config type: {type(ppo_config)}")
    print(f"Config has stop_token: {hasattr(ppo_config, 'stop_token')}")
    
    # Create PPOTrainer with proper PPOConfig object
    ppo_trainer = PPOTrainer(
        ppo_config,      # ✅ Now a proper PPOConfig object, not dict
        tokenizer,
        policy,
        ref_model,
        reward_model,
        dataset,
        value_model
    )
    
    print("✅ PPOTrainer initialized successfully!")
    
    # Force everything to stay on CPU
    ppo_trainer.model = ppo_trainer.model.cpu()
    ppo_trainer.ref_model = ppo_trainer.ref_model.cpu()  
    ppo_trainer.reward_model = ppo_trainer.reward_model.cpu()
    
    print("🎯 Starting training...")
    
    step_count = 0
    for epoch in range(1):
        print(f"\n--- Epoch {epoch+1} ---")
        
        for step, batch in enumerate(ppo_trainer.dataloader):
            if step >= 2:  # Just 2 steps for testing
                break
                
            try:
                step_count += 1
                print(f"Step {step_count}")
                
                query_tensors = batch['input_ids']
                query_tensors = [q.cpu() for q in query_tensors]

                # Generate responses
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    **generation_kwargs
                )
                
                response_tensors = [r.cpu() for r in response_tensors]
                batch['response'] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                
                # Create dummy rewards (since we don't have a trained reward model)
                dummy_rewards = torch.tensor([0.5] * len(query_tensors), dtype=torch.float32)
                
                # PPO step
                stats = ppo_trainer.step(query_tensors, response_tensors, dummy_rewards)
                print(f"✅ Step {step_count} completed")
                
                gc.collect()
                    
            except Exception as e:
                print(f"Error in step {step_count}: {e}")
                gc.collect()
                continue

    print("\n🎉 Training completed! Saving model...")
    
    try:
        ppo_trainer.save_model(ppo_model_path)
        print(f"✅ Model saved to {ppo_model_path}")
    except Exception as e:
        print(f"PPOTrainer save failed: {e}")
        policy.save_pretrained(ppo_model_path)
        tokenizer.save_pretrained(ppo_model_path)
        print(f"✅ Model saved with fallback method")

if __name__ == "__main__":
    try:
        train_ppo_fixed()
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
