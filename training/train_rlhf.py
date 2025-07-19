import torch
from transformers import AutoTokenizer, Phi3ForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from accelerate import infer_auto_device_map, dispatch_model
import gc
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def train_ppo(
    sft_model_path: str = "../data/models/sft_finetuned",
    reward_model_path: str = "../data/models/reward_model",
    ppo_model_path: str = "../data/models/ppo_policy",
    prompt_data_file: str = "../data/preferences.jsonl",
):
    ppo_config = PPOConfig(
        learning_rate=1.41e-5,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        stop_token_id=None,
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": None,
        "max_new_tokens": 64,
        "temperature": 0.7,
    }
   
    print("Loading models and tokenizer...")
   
    offload_dir = "./offload_cache"
    os.makedirs(offload_dir, exist_ok=True)
   
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
   
    ppo_config.stop_token_id = tokenizer.eos_token_id

    print("Loading policy model...")
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        sft_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    policy.gradient_checkpointing_enable()
    gc.collect()
    torch.cuda.empty_cache()
    
    device_map = infer_auto_device_map(
        policy,
        max_memory={0: "6GiB", "cpu": "10GiB"},
        no_split_module_classes=["Phi3DecoderLayer"]
    )
   
    policy = dispatch_model(
        policy,
        device_map=device_map,
        offload_dir=offload_dir
    )
   
    if not hasattr(policy, 'generation_config'):
        from transformers import GenerationConfig
        policy.generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=64,
        )
    else:
        policy.generation_config.eos_token_id = tokenizer.eos_token_id
        policy.generation_config.pad_token_id = tokenizer.pad_token_id

    gc.collect()
    torch.cuda.empty_cache()
    
    print("Loading reward model...")
    reward_model = Phi3ForSequenceClassification.from_pretrained(
        reward_model_path,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )

    reward_device_map = infer_auto_device_map(
        reward_model,
        max_memory={0: "6GiB", "cpu": "10GiB"},
        no_split_module_classes=["Phi3DecoderLayer"]
    )
   
    reward_model = dispatch_model(
        reward_model,
        device_map=reward_device_map,
        offload_dir=offload_dir
    )
   
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading and preparing dataset...")
   
    def tokenize(example):
        tokenized = tokenizer(example["prompt"], truncation=True, max_length=256)
        example["input_ids"] = tokenized["input_ids"]
        example["query"] = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)
        return example

    dataset = load_dataset("json", data_files=prompt_data_file, split="train")
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch", columns=["input_ids", "query"])

    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    print("Initializing PPOTrainer...")
   
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        train_dataset=dataset,
        value_model=None,
        data_collator=collator,
    )

    print("Starting PPO training loop...")
    for epoch in range(2):
        print(f"\n--- Epoch {epoch+1} ---")
        for batch in ppo_trainer.dataloader:
            query_tensors = batch['input_ids']

            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                **generation_kwargs
            )
            batch['response'] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
           
            texts = [q + r for q, r in zip(batch['query'], batch['response'])]
            reward_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
           
            reward_inputs = {k: v.to(next(reward_model.parameters()).device) for k, v in reward_inputs.items()}
           
            with torch.no_grad():
                rewards = reward_model(**reward_inputs).logits.squeeze(-1)
                rewards = rewards.cpu()
                torch.cuda.empty_cache()

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards.numpy())
            print(f"Mean PPO Reward: {stats['ppo/returns/mean']:.4f}")

    print("\nPPO training finished. Saving model...")
    ppo_trainer.save_model(ppo_model_path)
    print(f"PPO policy model saved to {ppo_model_path}")

if __name__ == "__main__":
    train_ppo()
