import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
# Load environment variables
load_dotenv()
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
# Page config
st.set_page_config(
    page_title="Chat with Nicolas Berndt",
    page_icon="🤖",
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
</style>
""", unsafe_allow_html=True)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hi! I'm Nicolas Berndt's AI assistant. Ask me anything about their content!"}
    ]
@st.cache_resource
def load_models():
    """Load FAISS index and models"""
    faiss_index = faiss.read_index("faiss_index")
    transcript_texts = np.load("texts.npy", allow_pickle=True)
    sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    return faiss_index, transcript_texts, sentence_transformer
def search_similar_transcripts(query, faiss_index, transcript_texts, sentence_transformer):
    """Search for relevant transcript segments"""
    query_vector = sentence_transformer.encode([query])[0]
    query_vector = np.array([query_vector]).astype('float32')
    
    k = 5
    distances, indices = faiss_index.search(query_vector, k)
    
    similar_texts = []
    for idx in indices[0]:
        similar_texts.append(transcript_texts[idx])
    
    return similar_texts
def generate_response(question, context):
    """Generate response using OpenAI API"""
    system_prompt = f"""You are an AI trained to respond exactly like Nicolas Berndt, based on their video transcripts. 
    Stay true to their style, knowledge, and way of explaining things. Use the provided transcript segments as your source of knowledge."""
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
            Based on these transcript segments:
            {context}
            
            Answer this question in Nicolas Berndt's style: {question}"""}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as error:
        return f"Sorry, I encountered an error: {str(error)}"
# Display creator header
st.markdown(
    f"""
    <div class="creator-header">
        <img src="https://yt3.ggpht.com/Nocpaj3XzTixS-Zy7uBjdjXmuAADyoVGzF1ACIJ_B3ic6uRxw1JqDfvytb7ZwVMonKXm-LkDjA=s240-c-k-c0x00ffffff-no-rj" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover;">
        <div class="creator-info">
            <h1 style="margin: 0;">Nicolas Berndt</h1>
            <p style="margin: 5px 0; color: #666;">
                435,000 subscribers • 70 videos
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
# Load models
try:
    faiss_index, transcript_texts, sentence_transformer = load_models()
except Exception as error:
    st.error(f"Error loading models: {str(error)}")
    st.stop()
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
# Chat input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
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