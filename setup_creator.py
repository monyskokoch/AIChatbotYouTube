import os
import argparse
from extract import TranscriptExtractor
from database import DatabaseCreator
import shutil
import json
from googleapiclient.discovery import build
import requests

class CreatorSetup:
    def __init__(self, channel_id, youtube_api_key, openai_api_key):
        self.channel_id = channel_id
        self.youtube_api_key = youtube_api_key
        self.openai_api_key = openai_api_key
        self.youtube = build("youtube", "v3", developerKey=youtube_api_key)
        
    def get_channel_info(self):
        """Get channel details from YouTube API"""
        try:
            request = self.youtube.channels().list(
                part="snippet,statistics",
                id=self.channel_id
            )
            response = request.execute()
            
            if 'items' in response:
                channel = response['items'][0]
                channel_name = channel['snippet']['title']
                
                # Create a clean version of the name for folder/file names
                clean_name = "".join(c for c in channel_name if c.isalnum() or c in (' ', '-', '_')).strip()
                clean_name = clean_name.replace(' ', '-').lower()
                
                info = {
                    'channel_id': self.channel_id,
                    'name': channel_name,
                    'clean_name': clean_name,
                    'description': channel['snippet']['description'],
                    'thumbnail_url': channel['snippet']['thumbnails']['medium']['url'],
                    'subscriber_count': channel['statistics']['subscriberCount'],
                    'video_count': channel['statistics']['videoCount']
                }
                
                return info
        except Exception as e:
            print(f"Error fetching channel info: {str(e)}")
            return None

    def setup_creator(self):
        """Set up a new creator bot"""
        try:
            print("\n=== Starting Creator Setup ===")
            
            # Step 1: Get Channel Info
            print("\n1️⃣ Fetching Channel Information...")
            channel_info = self.get_channel_info()
            if not channel_info:
                raise ValueError("Could not fetch channel information")
                
            creator_name = channel_info['clean_name']
            creator_dir = f"creators/{creator_name}"
            
            # Create creator directory
            os.makedirs(creator_dir, exist_ok=True)
            
            # Step 2: Extract Transcripts
            print("\n2️⃣ Extracting YouTube Transcripts...")
            extractor = TranscriptExtractor(self.youtube_api_key)
            csv_path = extractor.extract_for_channel(self.channel_id, output_dir=creator_dir)
            
            # Step 3: Create Database
            print("\n3️⃣ Creating Searchable Database...")
            db_creator = DatabaseCreator()
            faiss_path, texts_path = db_creator.create_for_channel(creator_dir)
            
            # Step 4: Create Deployment Directory
            print("\n4️⃣ Setting up Deployment...")
            deploy_dir = f"{creator_dir}/deployment"
            os.makedirs(deploy_dir, exist_ok=True)
            
            # Save channel info
            with open(f"{creator_dir}/info.json", 'w') as f:
                json.dump(channel_info, f, indent=4)
            
            # Create deployment files
            self.create_deployment_files(deploy_dir, channel_info)
            
            print(f"\n✨ Setup Complete for {channel_info['name']}! ✨")
            print(f"Deployment files created in: {deploy_dir}")
            print("\nTo deploy this creator's bot:")
            print("1. Create a new GitHub repository named: " + f"{creator_name}-bot")
            print("2. Copy the contents of the deployment directory there")
            print("3. Deploy to Streamlit Cloud using that repository")
            
            return creator_dir
            
        except Exception as e:
            print(f"\n❌ Error during setup: {str(e)}")
            raise

    def create_deployment_files(self, deploy_dir, channel_info):
        """Create all necessary files for deployment"""
        # Create streamlit app
        self.create_streamlit_app(deploy_dir, channel_info)
        
        # Copy necessary files
        shutil.copy2(f"{os.path.dirname(deploy_dir)}/faiss_index", f"{deploy_dir}/faiss_index")
        shutil.copy2(f"{os.path.dirname(deploy_dir)}/texts.npy", f"{deploy_dir}/texts.npy")
        
        # Create requirements.txt
        requirements = [
            "streamlit",
            "openai",
            "faiss-cpu",
            "numpy",
            "sentence-transformers",
            "python-dotenv"
        ]
        
        with open(f"{deploy_dir}/requirements.txt", 'w') as f:
            f.write('\n'.join(requirements))
            
        # Create README
        with open(f"{deploy_dir}/README.md", 'w') as f:
            f.write(f"# {channel_info['name']} AI Chatbot\n\n")
            f.write("AI-powered chatbot that lets you interact with content from ")
            f.write(f"{channel_info['name']}'s YouTube channel.\n")

    def create_streamlit_app(self, deploy_dir, channel_info):
        """Create the Streamlit app file"""
        app_content = f'''
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
    page_title="Chat with {channel_info['name']}",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .creator-header {{
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 20px;
    }}
    .creator-info {{
        flex: 1;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {{"role": "assistant", "content": f"Hi! I'm {channel_info['name']}'s AI assistant. Ask me anything about their content!"}}
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
    system_prompt = f"""You are an AI trained to respond exactly like {channel_info['name']}, based on their video transcripts. 
    Stay true to their style, knowledge, and way of explaining things. Use the provided transcript segments as your source of knowledge."""
    
    try:
        messages = [
            {{"role": "system", "content": system_prompt}},
            {{"role": "user", "content": f"""
            Based on these transcript segments:
            {{context}}
            
            Answer this question in {channel_info['name']}'s style: {{question}}"""}}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as error:
        return f"Sorry, I encountered an error: {{str(error)}}"

# Display creator header
st.markdown(
    f"""
    <div class="creator-header">
        <img src="{channel_info['thumbnail_url']}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover;">
        <div class="creator-info">
            <h1 style="margin: 0;">{channel_info['name']}</h1>
            <p style="margin: 5px 0; color: #666;">
                {int(channel_info['subscriber_count']):,} subscribers • {int(channel_info['video_count']):,} videos
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
    st.error(f"Error loading models: {{str(error)}}")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input():
    st.session_state.messages.append({{"role": "user", "content": prompt}})
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
            
            st.session_state.messages.append({{"role": "assistant", "content": response}})
'''
        
        with open(f"{deploy_dir}/streamlit_app.py", 'w') as f:
            f.write(app_content.strip())

def main():
    parser = argparse.ArgumentParser(description="Set up a creator's chatbot")
    parser.add_argument("--channel_id", required=True, help="YouTube channel ID")
    parser.add_argument("--youtube_key", required=True, help="YouTube API Key")
    parser.add_argument("--openai_key", required=True, help="OpenAI API Key")
    
    args = parser.parse_args()
    
    setup = CreatorSetup(
        channel_id=args.channel_id,
        youtube_api_key=args.youtube_key,
        openai_api_key=args.openai_key
    )
    
    setup.setup_creator()

if __name__ == "__main__":
    main()