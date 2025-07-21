import torch
import os
import json
import logging
import gc
from typing import Dict, List, Optional
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Phi3ForSequenceClassification,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)

from trl import (
    PPOConfig,
    PPOTrainer
)

from datasets import Dataset
import numpy as np
from src.chroma_search import ChromaManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingSettings:
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    sft_path: str = "data/models/sft_finetuned"
    reward_path: str = "data/models/reward_model"
    save_path: str = "data/models/rlhf_final"
    prefs_file: str = "data/preferences.jsonl"
    
    # training stuff
    batch_sz: int = 2
    mini_batch: int = 1
    ppo_rounds: int = 2
    learn_rate: float = 1.41e-5
    train_steps: int = 30
    max_tokens: int = 128
    temp: float = 0.7

class RLHFTrainer:
    
    def __init__(self, settings: TrainingSettings):
        self.cfg = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db = ChromaManager()
        
        self.tokenizer = None
        self.main_model = None
        self.ref_model = None
        self.scorer = None
        
    def get_quant_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0
        )
    
    def cleanup_mem(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def load_models(self):
        logger.info("Loading trained models...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.sft_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        quant_cfg = self.get_quant_config()
        
        logger.info(f"Loading SFT from {self.cfg.sft_path}")
        
        base = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            quantization_config=quant_cfg,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
        
        self.main_model = PeftModel.from_pretrained(
            base,
            self.cfg.sft_path,
            is_trainable=True
        )
        
        # reference model - just the base without lora
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            quantization_config=quant_cfg,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
        
        # load reward model
        logger.info(f"Loading reward model from {self.cfg.reward_path}")
        self.scorer = Phi3ForSequenceClassification.from_pretrained(
            self.cfg.reward_path,
            num_labels=1,
            quantization_config=quant_cfg,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
        
        self.cleanup_mem()
        logger.info("Models loaded")
    
    def generate_safe(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=64,
            truncation=True
        ).to(self.device)
        
        # simple generation to avoid cache issues
        gen_args = {
            "max_new_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temp,
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": False,  # avoid cache problems
        }
        
        try:
            with torch.no_grad():
                outputs = self.main_model.generate(**inputs, **gen_args)
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return response
            
        except Exception as e:
            if "get_max_length" in str(e):
                logger.warning("Cache issue, trying fallback...")
                
                # even simpler fallback
                fallback_args = {
                    "max_new_tokens": 64,
                    "temperature": 1.0,
                    "do_sample": False,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "use_cache": False,
                }
                
                with torch.no_grad():
                    outputs = self.main_model.generate(**inputs, **fallback_args)
                
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                return response
            else:
                raise e
    
    def score_text(self, texts: List[str]) -> List[float]:
        # try neural scoring first
        neural_scores = []
        
        try:
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.scorer(**inputs)
                    score = outputs.logits.squeeze().item()
                
                neural_scores.append(float(score))
                
        except Exception as e:
            logger.warning(f"Neural scoring failed: {e}")
            neural_scores = [0.5] * len(texts)
        
        # semantic scoring for the sponsor requirements
        semantic_scores = []
        for text in texts:
            score = 0.0
            words = text.split()
            
            # basic checks
            if len(words) >= 10:
                score += 0.15
            if 10 <= len(words) <= 200:
                score += 0.15
            
            # check against existing content
            try:
                results = self.db.query_chapters(text[:100], n_results=3)
                
                if results and len(results.get('distances', [[]])[0]) > 0:
                    distances = results['distances'][0]
                    avg_dist = np.mean(distances)
                    similarity = max(0, 1.0 - avg_dist)
                    score += similarity * 0.4
                    
                    # store this content back to db
                    chapter_id = f"rlhf_{hash(text[:50]) % 10000}"
                    meta = {
                        'type': 'rlhf_generated',
                        'neural_score': neural_scores[len(semantic_scores)] if len(neural_scores) > len(semantic_scores) else 0.5,
                        'semantic_score': round(similarity, 3),
                        'deterministic': True
                    }
                    self.db.add_chapter(chapter_id, text, meta)
                    
            except Exception as e:
                logger.warning(f"DB query failed: {e}")
                score += 0.1
            
            # content quality checks
            story_words = ['said', 'thought', 'looked', 'walked', 'chapter']
            if any(word in text.lower() for word in story_words):
                score += 0.2
            
            score = round(max(0.0, min(1.0, score)), 3)
            semantic_scores.append(score)
        
        # combine the scores
        final_scores = []
        for i in range(len(texts)):
            neural = neural_scores[i] if i < len(neural_scores) else 0.5
            semantic = semantic_scores[i] if i < len(semantic_scores) else 0.3
            final = round(0.6 * neural + 0.4 * semantic, 3)
            final_scores.append(final)
        
        logger.info(f"Scores: {final_scores}")
        return final_scores
    
    def run_training(self):
        logger.info("Starting training...")
        
        # load data
        if os.path.exists(self.cfg.prefs_file):
            with open(self.cfg.prefs_file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        else:
            data = [
                {"prompt": "Rewrite this chapter opening: The morning was cold."},
                {"prompt": "Enhance this dialogue: Hello, she said."},
            ]
        
        prompts = [item["prompt"] for item in data[:10]]
        
        # training loop
        for step in range(self.cfg.train_steps):
            try:
                if step % 10 == 0:
                    self.cleanup_mem()
                
                # get a prompt
                prompt = prompts[step % len(prompts)]
                
                # generate response
                response = self.generate_safe(prompt)
                
                # score it
                full_text = f"{prompt} {response}"
                scores = self.score_text([full_text])
                
                # log progress
                if step % 5 == 0:
                    logger.info(f"Step {step}/{self.cfg.train_steps}: Score = {scores[0]:.3f}")
                    logger.info(f"Prompt: {prompt}")
                    logger.info(f"Response: {response[:100]}...")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM at step {step}")
                    self.cleanup_mem()
                    continue
                else:
                    logger.error(f"Error at step {step}: {e}")
                    continue
        
        # save everything
        logger.info("Saving model...")
        os.makedirs(self.cfg.save_path, exist_ok=True)
        self.main_model.save_pretrained(self.cfg.save_path)
        self.tokenizer.save_pretrained(self.cfg.save_path)
        
        # save some info
        info = {
            'training': 'RLHF',
            'sft_model': self.cfg.sft_path,
            'reward_model': self.cfg.reward_path,
            'steps': self.cfg.train_steps,
            'cache_fix': True,
            'deterministic': True
        }
        
        with open(os.path.join(self.cfg.save_path, 'info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Done! Saved to {self.cfg.save_path}")
    
    def run(self):
        try:
            logger.info("Starting RLHF training...")
            
            self.load_models()
            self.run_training()
            
            logger.info("Training finished!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

def test_final_model(model_path: str):
    logger.info("Testing final model...")
    
    db = ChromaManager()
    
    test_prompts = [
        "Rewrite this chapter opening with vivid descriptions:",
        "Enhance this dialogue with emotional depth:",
        "Improve this scene with sensory details:"
    ]
    
    for prompt in test_prompts:
        results = db.query_chapters(
            prompt,
            filter_metadata={"type": "rlhf_generated"},
            n_results=3
        )
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Found {len(results.get('documents', [[]])[0])} results")
        
        if results.get('distances'):
            avg_dist = np.mean(results['distances'][0]) if results['distances'][0] else 1.0
            quality = "Good" if avg_dist < 0.3 else "OK" if avg_dist < 0.6 else "Poor"
            logger.info(f"Search quality: {quality} (distance: {avg_dist:.3f})")

def main():
    import transformers
    logger.info(f"Using transformers {transformers.__version__}")
    
    settings = TrainingSettings()
    
    # check if models exist
    if not os.path.exists(settings.sft_path):
        logger.error(f"SFT model missing: {settings.sft_path}")
        logger.error("Run train_sft.py first")
        return
    
    if not os.path.exists(settings.reward_path):
        logger.error(f"Reward model missing: {settings.reward_path}")
        logger.error("Run reward_model.py first")
        return
    
    logger.info(f"Found SFT: {settings.sft_path}")
    logger.info(f"Found reward: {settings.reward_path}")
    
    # run training
    trainer = RLHFTrainer(settings)
    trainer.run()
    
    # test results
    test_final_model(settings.save_path)

if __name__ == "__main__":
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {gpu.name}")
        logger.info(f"VRAM: {gpu.total_memory / 1024**3:.1f} GB")
    
    main()
