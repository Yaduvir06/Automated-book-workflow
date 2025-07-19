from flask import Flask, request, jsonify
from scraper import fetch_and_parse_chapter
from llm_utils import generate_text
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

        prompt = (
            "You are a creative and expert author. Rewrite the following chapter content with a more "
            "vivid, engaging, and modern writing style. Enhance descriptions and dialogue.\n\n"
            f"--- ORIGINAL CHAPTER CONTENT ---\n{truncated_text}\n\n"
            "--- REWRITTEN CHAPTER ---"
        )

        spun_chapter = generate_text(prompt)
        chap_id = str(uuid.uuid4())
        
        # Create metadata that aligns with sponsor's goals
        metadata = {
            'url': url,
            'version': 'ai-spun-v1',
            'status': 'draft', # Initial status
            'reward_score': 0.0, # Default score
            'screenshot_path': screenshot_path
        }
        db_manager.add_chapter(chapter_id=chap_id, text=spun_chapter, metadata=metadata)

        # Run TTS in a separate thread to avoid blocking the response
        tts_thread = threading.Thread(target=speak_text, args=(spun_chapter[:250],))
        tts_thread.start()

        return jsonify({'id': chap_id, 'spun_chapter': spun_chapter, 'original_url': url})

    except Exception as e:
        logging.error(f"An error occurred in /spin endpoint: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        feedback_data = request.json
        if not all(key in feedback_data for key in ["prompt", "good", "bad"]):
            return jsonify({'error': 'Invalid feedback format.'}), 400

        with open(PREF_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + '\n')
        
        return jsonify({'status': 'Feedback saved successfully'})

    except Exception as e:
        logging.error(f"An error occurred in /feedback endpoint: {e}", exc_info=True)
        return jsonify({'error': 'Failed to save feedback.'}), 500

@app.route('/search', methods=['GET'])
def search_chapters():
    try:
        query = request.args.get('q')
        status = request.args.get('status') # For deterministic search
        if not query:
            return jsonify({'error': 'Query parameter "q" is required.'}), 400
        
        filter_metadata = {}
        if status:
            filter_metadata["status"] = status
            logging.info(f"Performing search for '{query}' with status filter: '{status}'")
        else:
            logging.info(f"Performing search for query: '{query}'")

        results = db_manager.query_chapters(query, filter_metadata=filter_metadata or None)
        return jsonify(results)

    except Exception as e:
        logging.error(f"An error occurred in /search endpoint: {e}", exc_info=True)
        return jsonify({'error': 'Search failed.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

