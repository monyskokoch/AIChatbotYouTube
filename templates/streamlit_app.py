import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import time
import tiktoken
import json
from typing import List
import random

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key-here")

# Load creator info
def load_creator_info():
    with open("creator_info.json", "r") as f:
        return json.load(f)

# Page config
creator_info = load_creator_info()
st.set_page_config(
    page_title=f"Chat with {creator_info['name']}",
    page_icon="🎭",
    layout="centered"
)

# Custom CSS
def apply_custom_css():
    st.markdown("""
        <style>
        /* Modern Color Scheme */
        :root {
            --primary-color: #2D2D2D;
            --secondary-color: #F8F9FA;
            --accent-color: #FF4B4B;
            --text-color: #333333;
            --light-gray: #E9ECEF;
        }
        
        /* Main Container */
        .main {
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Creator Header */
        .creator-header {
            background: linear-gradient(135deg, var(--secondary-color) 0%, #FFFFFF 100%);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            display: flex;
            align-items: center;
            gap: 2rem;
            transition: transform 0.2s ease;
        }
        
        .creator-header:hover {
            transform: translateY(-2px);
        }
        
        .creator-avatar {
            width: 120px;
            height: 120px;
            border-radius: 60px;
            object-fit: cover;
            border: 4px solid white;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .creator-info {
            flex: 1;
        }
        
        .creator-name {
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            color: var(--primary-color);
        }
        
        .creator-stats {
            color: #666;
            font-size: 1rem;
            margin-top: 0.5rem;
        }

        /* Suggested Questions */
        .suggested-questions {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin: 2rem 0;
        }

        .question-chip {
            background: var(--light-gray);
            padding: 0.8rem 1.5rem;
            border-radius: 25px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
            color: var(--text-color);
        }

        .question-chip:hover {
            background: var(--accent-color);
            color: white;
            transform: translateY(-2px);
        }
        
        /* Chat Messages */
        .chat-message {
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            animation: fadeIn 0.3s ease;
            max-width: 85%;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        [data-testid="stChatMessage"] {
            background: var(--light-gray);
            border-radius: 15px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        /* User messages */
        [data-testid="stChatMessage"][data-chat-message-user-name="user"] {
            background: var(--accent-color);
            color: white;
            margin-left: 15%;
        }
        
        /* Assistant messages */
        [data-testid="stChatMessage"][data-chat-message-user-name="assistant"] {
            background: var(--secondary-color);
            margin-right: 15%;
        }
        
        /* Input Area */
        [data-testid="stChatInput"] {
            border-radius: 25px !important;
            border: 2px solid var(--light-gray) !important;
            padding: 0.8rem 1.5rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        }
        
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--accent-color) !important;
            box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.1) !important;
        }
        
        /* Spinner */
        .stSpinner {
            border-color: var(--accent-color) !important;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .creator-header {
                flex-direction: column;
                text-align: center;
                padding: 1.5rem;
            }
            
            .creator-avatar {
                width: 100px;
                height: 100px;
            }
            
            .chat-message {
                margin-left: 1rem;
                margin-right: 1rem;
                max-width: 90%;
            }
        }
        </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hi! I'm {creator_info['name']}'s AI assistant. Ask me anything about their content!"}
    ]

# Token counter
def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# Smart text chunker
def smart_chunk_text(text: str, max_tokens: int = 1000) -> list:
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current_length + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

# Load the necessary files
@st.cache_resource
def load_models():
    faiss_index = faiss.read_index("faiss_index")
    transcript_texts = np.load("texts.npy", allow_pickle=True)
    sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    return faiss_index, transcript_texts, sentence_transformer

# Generate suggested questions based on transcripts
def generate_suggested_questions(transcript_texts: List[str], n: int = 5) -> List[str]:
    """Generate relevant questions based on actual transcript content."""
    try:
        # Combine some random transcripts for analysis
        sample_size = min(20, len(transcript_texts))
        text_sample = random.sample(list(transcript_texts), sample_size)
        combined_text = " ".join(text_sample)
        
        # Use GPT to analyze content and generate relevant questions
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are an expert at analyzing content and generating engaging questions. 
                Given some transcript text, generate natural, specific questions that viewers would want to ask about this content.
                Questions should:
                1. Be specific to the actual content
                2. Use natural language
                3. Focus on interesting topics/opinions from the content
                4. Be varied in type (how, what, why questions)
                5. Avoid generic templates"""},
                {"role": "user", "content": f"""Based on these transcript segments, generate {n} natural questions that viewers would want to ask.
                Only generate questions about topics actually discussed in the content.
                Format as a simple list of questions.
                
                Transcript samples:
                {combined_text}"""}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        # Process response into list
        questions_text = response.choices[0].message.content
        questions = [q.strip().strip('*-.1234567890') for q in questions_text.split('\n') if q.strip()]
        
        return questions[:n]
        
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        # Fallback to basic questions if something goes wrong
        return [
            "What's your most popular video about?",
            "How did you get started?",
            "What's your best advice?",
            "What's your creative process?",
            "What inspired you?"
        ]

def search_similar_transcripts(query, faiss_index, transcript_texts, sentence_transformer):
    """Search for relevant transcript segments with token limiting"""
    query_vector = sentence_transformer.encode([query])[0]
    query_vector = np.array([query_vector]).astype('float32')
    
    k = 3  # Reduced from 5 to limit context
    distances, indices = faiss_index.search(query_vector, k)
    
    similar_texts = []
    total_length = 0
    max_chars = 3000  # Approximate token limit based on characters
    
    for idx in indices[0]:
        text = transcript_texts[idx]
        if total_length + len(text) <= max_chars:
            similar_texts.append(text)
            total_length += len(text)
        else:
            break
    
    return similar_texts

def generate_response(question, context):
    """Generate response with comprehensive error handling and fallback mechanisms"""
    system_prompt = f"""You are an AI trained to respond exactly like {creator_info['name']}, based on their video transcripts. 
    Stay true to their style, knowledge, and way of explaining things. Use the provided transcript segments as your source of knowledge.
    Be extremely specific and personal in your responses, as if {creator_info['name']} is directly speaking to their audience."""
    
    # Fallback response function
    def get_fallback_response():
        return f"I apologize, but I'm having trouble processing this specific query about {creator_info['name']}'s content. Could you rephrase your question or try a different approach?"
    
    try:
        # Intelligent context management
        def prepare_context(context, max_tokens=3000):
            # Convert context to string and truncate
            combined_context = " ".join(context)
            
            # Use tiktoken to accurately count tokens
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(combined_context)
            
            # Truncate if over token limit
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                combined_context = encoding.decode(truncated_tokens)
            
            return combined_context
        
        # Prepare context
        truncated_context = prepare_context(context)
        
        # Prepare messages with intelligent token management
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            Based on these transcript segments:
            {truncated_context}
            
            Answer this question in {creator_info['name']}'s style: {question}"""}
        ]
        
        # Retry mechanism with multiple models and strategies
        models_to_try = [
            {"model": "gpt-4", "max_tokens": 500, "temperature": 0.7},
            {"model": "gpt-3.5-turbo", "max_tokens": 300, "temperature": 0.7},
        ]
        
        for model_config in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_config['model'],
                    messages=messages,
                    temperature=model_config['temperature'],
                    max_tokens=model_config['max_tokens']
                )
                return response.choices[0].message.content
            
            except Exception as model_error:
                # Log the specific model error
                print(f"Error with {model_config['model']}: {str(model_error)}")
                
                # If it's a token limit error, reduce context further
                if "context_length_exceeded" in str(model_error):
                    messages[1]['content'] = f"""
                    Summarized transcript segments:
                    {truncated_context[:1000]}
                    
                    Concisely answer this question in {creator_info['name']}'s style: {question}"""
                    continue
                
                # If it's a rate limit error, wait and retry
                if "rate_limit" in str(model_error).lower():
                    time.sleep(30)
                    continue
        
        # If all model attempts fail, return fallback
        return get_fallback_response()
                
    except Exception as error:
        # Comprehensive error logging
        print(f"Unexpected error in generate_response: {str(error)}")
        return get_fallback_response()

# Handle suggested question click
def handle_question_click(question):
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            faiss_index, transcript_texts, sentence_transformer = load_models()
            similar_texts = search_similar_transcripts(
                question, 
                faiss_index, 
                transcript_texts, 
                sentence_transformer
            )
            
            response = generate_response(question, similar_texts)
            
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Main chat interface
def main():
    # Apply custom CSS
    apply_custom_css()
    
    # Rest of your main function stays the same
    st.markdown(
        f"""
        <div class="creator-header">
            <img src="assets/profile.jpg" class="creator-avatar">
            <div class="creator-info">
                <h1 class="creator-name">{creator_info['name']}</h1>
                <p class="creator-description">{creator_info['description'][:150]}...</p>
                <div class="creator-stats">
                    {int(creator_info['subscriber_count']):,} subscribers • {int(creator_info['video_count']):,} videos
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Load models
    try:
        faiss_index, transcript_texts, sentence_transformer = load_models()
    except Exception as e:
        st.error("Error loading models. Please try again later.")
        return

    # Suggested questions
    st.markdown("### 💭 Suggested Questions")
    questions = generate_suggested_questions(transcript_texts)
    
    # Display questions as clickable chips
    st.markdown('<div class="suggested-questions">', unsafe_allow_html=True)
    cols = st.columns(len(questions))
    for i, question in enumerate(questions):
        if cols[i].button(question, key=f"q_{i}"):
            handle_question_click(question)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input(f"Ask anything about {creator_info['name']}'s content"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                similar_texts = search_similar_transcripts(
                    prompt, 
                    faiss_index, 
                    transcript_texts, 
                    sentence_transformer
                )
                
                response = generate_response(prompt, similar_texts)
                
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()


