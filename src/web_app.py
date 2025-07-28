from flask import Flask, request, jsonify
from scraper import fetch_and_parse_chapter
from llm_utils import generate_text, tokenizer, model
from chroma_search import ChromaManager # <-- Use the manager class
from tts_module import speak_text
import uuid
import json
import os
import logging
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# --- Setup Data Directory, Feedback File, and ChromaDB ---
DATA_DIR = '../data'
PREF_FILE = os.path.join(DATA_DIR, 'preferences.jsonl')
os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(PREF_FILE):
    open(PREF_FILE, 'w').close()
    logging.info(f"Created empty preferences file at {PREF_FILE}")

# Initialize ChromaDB Manager
db_manager = ChromaManager()

# Log model information at startup
def log_model_info():
    """Log information about the loaded model"""
    from peft import PeftModel
    if isinstance(model, PeftModel):
        logging.info(" Using RLHF-trained PEFT model")
        logging.info(f"Base model: {model.base_model.config.name_or_path if hasattr(model.base_model, 'config') else 'Unknown'}")
        logging.info(f"PEFT config: {type(model.peft_config) if hasattr(model, 'peft_config') else 'Unknown'}")
    else:
        logging.info(" Using base model (RLHF model not loaded)")
        logging.info(f"Model: {model.config.name_or_path if hasattr(model, 'config') else 'Unknown'}")

# Log model info at startup
log_model_info()

@app.route('/spin', methods=['POST'])
def spin_chapter():
    try:
        data = request.json
        url = data.get('url')
        if not url:
            return jsonify({'error': 'URL is required'}), 400

        logging.info(f"Spin request received for URL: {url}")
        clean_text, screenshot_path = fetch_and_parse_chapter(url)
        truncated_text = clean_text[:3000] 

        # Enhanced prompt for RLHF model
        prompt = (
            "You are a creative and expert author. Rewrite the following chapter content with a more "
            "vivid, engaging, and modern writing style. Enhance descriptions and dialogue. "
            "Focus on creating compelling narrative flow and character development.\n\n"
            f"--- ORIGINAL CHAPTER CONTENT ---\n{truncated_text}\n\n"
            "--- REWRITTEN CHAPTER ---"
        )

        try:
            spun_chapter = generate_text(prompt, max_length=512)
            logging.info("Text generation successful")
        except Exception as gen_error:
            logging.error(f"Text generation failed: {gen_error}")
            spun_chapter = f"Error generating text: {str(gen_error)}"

        chap_id = str(uuid.uuid4())
        
        # Create metadata that aligns with sponsor's goals
        metadata = {
            'url': url,
            'version': 'ai-spun-rlhf-v1',  # Updated to indicate RLHF usage
            'status': 'draft', # Initial status
            'reward_score': 0.0, # Default score
            'screenshot_path': screenshot_path,
            'model_type': 'rlhf' if hasattr(model, 'peft_config') else 'base'
        }
        
        try:
            db_manager.add_chapter(chapter_id=chap_id, text=spun_chapter, metadata=metadata)
            logging.info(" Chapter added to ChromaDB")
        except Exception as db_error:
            logging.error(f"ChromaDB insertion failed: {db_error}")

        # Run TTS in a separate thread to avoid blocking the response
        try:
            tts_thread = threading.Thread(target=speak_text, args=(spun_chapter[:250],))
            tts_thread.start()
            logging.info("TTS thread started")
        except Exception as tts_error:
            logging.error(f"TTS failed: {tts_error}")

        return jsonify({
            'id': chap_id, 
            'spun_chapter': spun_chapter, 
            'original_url': url,
            'model_type': metadata['model_type']
        })

    except Exception as e:
        logging.error(f"An error occurred in /spin endpoint: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        feedback_data = request.json
        if not all(key in feedback_data for key in ["prompt", "good", "bad"]):
            return jsonify({'error': 'Invalid feedback format.'}), 400

        # Add model type to feedback for analysis
        feedback_data['model_type'] = 'rlhf' if hasattr(model, 'peft_config') else 'base'
        feedback_data['timestamp'] = str(uuid.uuid4())  # Add unique identifier

        with open(PREF_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + '\n')
        
        logging.info(f"Feedback saved for {feedback_data['model_type']} model")
        return jsonify({'status': 'Feedback saved successfully'})

    except Exception as e:
        logging.error(f"An error occurred in /feedback endpoint: {e}", exc_info=True)
        return jsonify({'error': 'Failed to save feedback.'}), 500

@app.route('/search', methods=['GET'])
def search_chapters():
    try:
        query = request.args.get('q')
        status = request.args.get('status') # For deterministic search
        model_type = request.args.get('model_type')  # New parameter for filtering by model type
        
        if not query:
            return jsonify({'error': 'Query parameter "q" is required.'}), 400
        
        filter_metadata = {}
        if status:
            filter_metadata["status"] = status
        if model_type:
            filter_metadata["model_type"] = model_type
            
        if filter_metadata:
            logging.info(f"Performing search for '{query}' with filters: {filter_metadata}")
        else:
            logging.info(f"Performing search for query: '{query}'")

        try:
            results = db_manager.query_chapters(query, filter_metadata=filter_metadata or None)
            logging.info(f"Search returned {len(results.get('documents', [[]])[0])} results")
            return jsonify(results)
        except Exception as search_error:
            logging.error(f"ChromaDB search failed: {search_error}")
            return jsonify({'error': 'Search failed due to database error.'}), 500

    except Exception as e:
        logging.error(f"An error occurred in /search endpoint: {e}", exc_info=True)
        return jsonify({'error': 'Search failed.'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """New endpoint to get information about the currently loaded model"""
    try:
        from peft import PeftModel
        info = {
            'is_rlhf_model': isinstance(model, PeftModel),
            'model_type': 'rlhf' if isinstance(model, PeftModel) else 'base',
            'tokenizer_vocab_size': len(tokenizer),
            'model_device': str(model.device) if hasattr(model, 'device') else 'Unknown'
        }
        
        if isinstance(model, PeftModel):
            info['base_model_name'] = getattr(model.base_model.config, 'name_or_path', 'Unknown')
            info['peft_type'] = type(model.peft_config).__name__ if hasattr(model, 'peft_config') else 'Unknown'
        else:
            info['model_name'] = getattr(model.config, 'name_or_path', 'Unknown')
        
        return jsonify(info)
    except Exception as e:
        logging.error(f"Error getting model info: {e}")
        return jsonify({'error': 'Could not retrieve model information'}), 500

if __name__ == '__main__':
    logging.info(" Starting Flask application...")
    logging.info(f"Using model: {'RLHF-trained' if hasattr(model, 'peft_config') else 'Base Phi-3'}")
    app.run(host='0.0.0.0', port=5000, debug=False)
