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
st.markdown("""
<style>
    .creator-header {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .creator-info {
        flex: 1;
    }
    .creator-stats {
        font-size: 0.9em;
        color: #666;
    }
    .suggested-questions {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 20px 0;
    }
    .question-chip {
        background: #f0f2f6;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 0.9em;
        cursor: pointer;
        transition: background 0.3s;
    }
    .question-chip:hover {
        background: #e0e2e6;
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
    """Generate response with rate limit handling"""
    system_prompt = f"""You are an AI trained to respond exactly like {channel_info['name']}, based on their video transcripts. 
    Stay true to their style, knowledge, and way of explaining things. Use the provided transcript segments as your source of knowledge.
    Be extremely specific and personal in your responses, as if {channel_info['name']} is directly speaking to their audience."""
    
    try:
        # Join context with length limit
        combined_context = " ".join(context)
        if len(combined_context) > 3000:
            combined_context = combined_context[:3000] + "..."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            Based on these transcript segments:
            {combined_context}
            
            Answer this question in {channel_info['name']}'s style: {question}"""}
        ]
        
        # Add retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(20)  # Wait before retrying
                    continue
                raise
                
    except Exception as error:
        return f"Sorry, I encountered an error: {str(error)}"

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
    # Creator header
    st.markdown(
        f"""
        <div class="creator-header">
            <img src="assets/profile.jpg" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover;">
            <div class="creator-info">
                <h2 style="margin: 0;">{creator_info['name']}</h2>
                <p style="margin: 5px 0;">{creator_info['description'][:150]}...</p>
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


