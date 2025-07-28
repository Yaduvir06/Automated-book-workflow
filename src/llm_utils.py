from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_quantization_config():
    """Get quantization config matching your training setup"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0
    )

def load_rlhf_model(model_path: str = "../data/models/rlhf_final"):
    """Load RLHF-trained PEFT model"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"RLHF model not found at {model_path}")
        logger.info("Falling back to base model...")
        return load_base_model()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Loading model on CPU (will be slow).")
        device_map = None
        quantization_config = None
    else:
        device_map = {"": 0}
        quantization_config = get_quantization_config()
        logger.info(f"Loading model on {torch.cuda.get_device_name(0)}")
    
    try:
        # Load tokenizer from RLHF model directory
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # Load base model first
        base_model_name = "microsoft/Phi-3-mini-4k-instruct"
        logger.info(f"Loading base model: {base_model_name}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
        
        # Load PEFT model
        logger.info(f"Loading RLHF PEFT model from {model_path}")
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            is_trainable=False  # Set to False for inference
        )
        
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(" RLHF model loaded successfully")
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Failed to load RLHF model: {e}")
        logger.info("Falling back to base model...")
        return load_base_model()

def load_base_model(model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
    """Load base Phi-3 model without PEFT (fallback)"""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Loading model on CPU (will be slow).")
        device_map = None
        quantization_config = None
    else:
        device_map = "auto"
        quantization_config = get_quantization_config()
        logger.info(f"Loading model on {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16, 
        trust_remote_code=True,
        device_map=device_map,      
        low_cpu_mem_usage=True,
        attn_implementation="eager"  
    )
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Base model loaded successfully")
    return tokenizer, model

def load_standard_model(model_name: str = "../data/models/rlhf_final"):
    """Load model - tries RLHF first, falls back to base model"""
    return load_rlhf_model(model_name)

# Initialize the model
tokenizer, model = load_standard_model()

def generate_text(prompt: str, max_length: int = 512) -> str:
    """Generate text using the loaded model (RLHF or base)"""
    
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
    
    # Generation parameters that avoid cache issues
    generation_kwargs = {
        "max_new_tokens": max_length,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,
        "use_cache": False,  # Disable cache to avoid DynamicCache issues
        "return_dict_in_generate": False
    }
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Decode only the new tokens
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        # Fallback with even simpler parameters
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(max_length, 128),
                    temperature=1.0,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
            
            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e2:
            logger.error(f"Fallback generation also failed: {e2}")
            return "Error: Could not generate text"
