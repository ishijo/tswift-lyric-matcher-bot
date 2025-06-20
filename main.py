
from flask import Flask, render_template, request, jsonify
import os
import glob
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Initialize the AI model for text similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Store lyrics and their embeddings
lyrics_database = {}
lyrics_embeddings = {}

def load_lyrics_files():
    """Load all .txt files from the lyrics directory and subdirectories and compute embeddings"""
    lyrics_dir = 'lyrics'
    if not os.path.exists(lyrics_dir):
        os.makedirs(lyrics_dir)
        return
    
    # Recursively find all .txt files in lyrics directory and subdirectories
    txt_files = glob.glob(os.path.join(lyrics_dir, '**', '*.txt'), recursive=True)
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    # Get relative path from lyrics directory for better song identification
                    rel_path = os.path.relpath(file_path, lyrics_dir)
                    album_folder = os.path.dirname(rel_path)
                    song_filename = os.path.basename(file_path).replace('.txt', '')
                    
                    # Create song name with album context if in subfolder
                    if album_folder and album_folder != '.':
                        song_name = f"{album_folder}/{song_filename}"
                    else:
                        song_name = song_filename
                    
                    lyrics_database[song_name] = content
                    
                    # Split lyrics into chunks for better matching
                    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                    if chunks:
                        embeddings = model.encode(chunks)
                        lyrics_embeddings[song_name] = {
                            'chunks': chunks,
                            'embeddings': embeddings
                        }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

def find_best_matching_lyrics(scenario, top_k=3):
    """Find the most relevant lyrics for a given scenario"""
    if not lyrics_embeddings:
        return []
    
    scenario_embedding = model.encode([scenario])
    
    all_matches = []
    
    for song_name, data in lyrics_embeddings.items():
        similarities = cosine_similarity(scenario_embedding, data['embeddings'])[0]
        
        # Get top matches for this song
        top_indices = np.argsort(similarities)[-2:][::-1]  # Top 2 chunks per song
        
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                all_matches.append({
                    'song': song_name,
                    'lyrics': data['chunks'][idx],
                    'similarity': float(similarities[idx])
                })
    
    # Sort by similarity and return top matches
    all_matches.sort(key=lambda x: x['similarity'], reverse=True)
    return all_matches[:top_k]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_lyrics', methods=['POST'])
def find_lyrics():
    try:
        data = request.get_json()
        scenario = data.get('scenario', '').strip()
        
        if not scenario:
            return jsonify({'error': 'Please provide a life scenario'}), 400
        
        # Load lyrics if not already loaded
        if not lyrics_database:
            load_lyrics_files()
        
        if not lyrics_database:
            return jsonify({'error': 'No lyrics files found. Please add .txt files to the lyrics folder.'}), 400
        
        matches = find_best_matching_lyrics(scenario)
        
        if not matches:
            return jsonify({'message': 'No matching lyrics found for your scenario.'})
        
        return jsonify({'matches': matches})
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/reload_lyrics', methods=['POST'])
def reload_lyrics():
    """Reload lyrics files (useful when new files are added)"""
    try:
        global lyrics_database, lyrics_embeddings
        lyrics_database = {}
        lyrics_embeddings = {}
        load_lyrics_files()
        
        return jsonify({
            'message': f'Loaded {len(lyrics_database)} songs successfully',
            'songs': list(lyrics_database.keys())
        })
    except Exception as e:
        return jsonify({'error': f'Error reloading lyrics: {str(e)}'}), 500

if __name__ == '__main__':
    # Load lyrics on startup
    load_lyrics_files()
    app.run(host='0.0.0.0', port=5000, debug=True)
