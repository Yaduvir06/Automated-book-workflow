from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_standard_model(model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
    """Load Phi-3 model without quantization'"""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Loading model on CPU (will be slow).")
        device_map = None
    else:
        device_map = "auto"
        logger.info(f"Loading model on {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, 
        trust_remote_code=True,
        device_map=device_map,      
        low_cpu_mem_usage=True,
        attn_implementation="eager"  
    )
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Model loaded successfully without quantization")
    return tokenizer, model


tokenizer, model = load_standard_model()

def generate_text(prompt: str, max_length: int = 512) -> str:
    """Generate text using the standard Phi-3 model"""
    
    # Format prompt for Phi-3
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )
    
    # Move inputs to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            use_cache=True  
        )
    
    # Decode only the new tokens
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated_text.strip()
