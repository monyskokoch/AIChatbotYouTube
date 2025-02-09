from flask import Flask, request, jsonify
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import Tuple, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor
import re
from collections import Counter

app = Flask(__name__)

# Initialize global variables
faiss_index = None
transcript_texts = None
sentence_transformer = None
executor = ThreadPoolExecutor(max_workers=3)

class ChatbotConfig:
    FAISS_INDEX_PATH = "youtube_faiss_index"
    TRANSCRIPT_PATH = "youtube_texts.npy"
    MODEL_NAME = "all-MiniLM-L6-v2"
    TOP_K_RESULTS = 15
    MIN_TRANSCRIPT_LENGTH = 50
    MAX_TOTAL_TOKENS = 3000
    MAX_SEARCH_QUERIES = 20

    SYSTEM_PROMPT = """You must provide responses that EXACTLY match the content creator's demonstrated opinions, style, and advice from their transcripts, with no modifications or filtering.

    Core Requirements:
    1. Absolute Content Loyalty:
       - Maintain ALL of the creator's opinions and viewpoints exactly as expressed
       - Never soften, modify, or filter their positions
       - Keep their exact attitude, even if controversial
       - Preserve their level of directness/harshness
       
    2. Style Accuracy:
       - Use their exact manner of speaking
       - Keep their tone (whether arrogant, humble, harsh, etc.)
       - Maintain their way of addressing others
       - Use their typical phrases and expressions
       
    3. Response Generation:
       - Base ALL advice purely on their transcript content
       - Don't add moral disclaimers or soften opinions
       - Stay true to their approach and recommendations
       - Keep their level of respect/disrespect
       
    4. Format Considerations:
       - Focus on chat-appropriate responses (no "watch my videos")
       - Don't include video-specific elements (no "link in bio")
       - Maintain natural conversation flow
       - End responses appropriately for chat

    CRITICAL: You are NOT to modify, soften, or ethically filter ANY of their positions or style.
    Your ONLY goal is to provide responses that perfectly match what this creator would say based
    on their transcript content."""

def init_chatbot() -> None:
    """Initialize the chatbot components."""
    global faiss_index, transcript_texts, sentence_transformer
    
    try:
        faiss_index = faiss.read_index(ChatbotConfig.FAISS_INDEX_PATH)
        transcript_texts = np.load(ChatbotConfig.TRANSCRIPT_PATH, allow_pickle=True)
        sentence_transformer = SentenceTransformer(ChatbotConfig.MODEL_NAME)
        print("Chatbot initialization successful!")
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        raise

def get_semantic_search_queries(question: str) -> List[str]:
    """
    Generate search queries based on the question without predetermined topics.
    Focuses on extracting meaningful phrases and variations.
    """
    queries = [question]  # Original question is always included
    
    # Clean and tokenize the question
    cleaned_question = re.sub(r'[^\w\s]', ' ', question.lower())
    words = cleaned_question.split()
    
    # Add individual important words (excluding stop words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
    important_words = [w for w in words if w not in stop_words]
    queries.extend(important_words)
    
    # Add consecutive word pairs (bigrams)
    if len(words) > 1:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        queries.extend(bigrams)
    
    # Add triplets for longer phrases
    if len(words) > 2:
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
        queries.extend(trigrams)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for query in queries:
        if query not in seen:
            seen.add(query)
            unique_queries.append(query)
    
    return unique_queries[:ChatbotConfig.MAX_SEARCH_QUERIES]

def search_similar_transcripts(question: str) -> Tuple[list, list]:
    """
    Search for similar transcripts using multiple search strategies.
    """
    all_relevant_texts = []
    all_distances = []
    seen_texts = set()
    
    # Get search queries
    search_queries = get_semantic_search_queries(question)
    
    for search_query in search_queries:
        query_vector = sentence_transformer.encode([search_query])[0]
        query_vector = np.array([query_vector]).astype('float32')
        
        # Search for each query
        distances, indices = faiss_index.search(query_vector, ChatbotConfig.TOP_K_RESULTS)
        
        # Process results with context windows
        for idx, distance in zip(indices[0], distances[0]):
            # Get context window
            start_idx = max(0, idx - 2)
            end_idx = min(len(transcript_texts), idx + 3)
            
            # Add context segments
            for i in range(start_idx, end_idx):
                transcript = transcript_texts[i]
                
                # Only add if it's not a duplicate and meets length requirement
                if transcript not in seen_texts and len(transcript) >= ChatbotConfig.MIN_TRANSCRIPT_LENGTH:
                    all_relevant_texts.append(transcript)
                    all_distances.append(distance)
                    seen_texts.add(transcript)

    return all_relevant_texts, all_distances

def select_best_transcripts(transcripts: List[str], max_tokens: int) -> List[str]:
    """
    Select the most relevant and diverse transcript segments within token limit.
    """
    if not transcripts:
        return []

    selected_transcripts = []
    current_tokens = 0
    
    for transcript in transcripts:
        estimated_tokens = len(transcript.split()) * 1.3
        
        if current_tokens + estimated_tokens <= max_tokens:
            selected_transcripts.append(transcript)
            current_tokens += estimated_tokens
    
    return selected_transcripts

def generate_response(question: str, context: list) -> str:
    """
    Generate response using OpenAI's API with comprehensive context.
    """
    try:
        selected_transcripts = select_best_transcripts(context, ChatbotConfig.MAX_TOTAL_TOKENS)
        
        if not selected_transcripts:
            return "I couldn't find enough relevant content to provide a good response. Could you try rephrasing your question?"
        
        combined_context = "\n\n=== TRANSCRIPT SEGMENT ===\n".join(selected_transcripts)
        
        messages = [
            {"role": "system", "content": ChatbotConfig.SYSTEM_PROMPT},
            {"role": "user", "content": f"""
            Review these transcript segments from multiple videos by the content creator:
            
            {combined_context}
            
            Based on the creator's style, perspectives, and patterns shown across
            these different videos, please provide a comprehensive answer to this question: {question}
            
            Remember to synthesize insights from multiple segments and maintain their authentic voice.
            """}
        ]
        
        client = app.config['OPENAI_CLIENT']
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            presence_penalty=0.6
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return None

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat endpoint."""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        if 'message' not in data:
            return jsonify({"error": "Missing 'message' field"}), 400
            
        question = data['message']
        
        similar_texts, distances = search_similar_transcripts(question)
        
        if not similar_texts:
            return jsonify({
                "response": "I couldn't find enough relevant content to provide a good response. Could you try rephrasing your question?"
            })
        
        response = generate_response(question, similar_texts)
        if response is None:
            return jsonify({"error": "Failed to generate response"}), 500
            
        return jsonify({"response": response})
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

def main():
    """Main entry point."""
    try:
        init_chatbot()
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', 'sk-dummy-key'))  # Replace with your actual API key
        app.config['OPENAI_CLIENT'] = client
        app.run(host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"Failed to start the chatbot: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()