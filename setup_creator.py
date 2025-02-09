import os
import argparse
from extract import TranscriptExtractor
from database import DatabaseCreator
import shutil
import subprocess
import json
from googleapiclient.discovery import build
import requests

class CreatorSetup:
    def __init__(self, channel_id, youtube_api_key, openai_api_key):
        self.channel_id = channel_id
        self.youtube_api_key = youtube_api_key
        self.openai_api_key = openai_api_key
        self.creator_dir = f"creators/{channel_id}"
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
                info = {
                    'name': channel['snippet']['title'],
                    'description': channel['snippet']['description'],
                    'thumbnail_url': channel['snippet']['thumbnails']['medium']['url'],
                    'subscriber_count': channel['statistics']['subscriberCount'],
                    'video_count': channel['statistics']['videoCount']
                }
                
                # Download thumbnail
                thumbnail_response = requests.get(info['thumbnail_url'])
                if thumbnail_response.status_code == 200:
                    os.makedirs(f"{self.creator_dir}/assets", exist_ok=True)
                    with open(f"{self.creator_dir}/assets/profile.jpg", 'wb') as f:
                        f.write(thumbnail_response.content)
                
                return info
        except Exception as e:
            print(f"Error fetching channel info: {str(e)}")
            return None

    def run_setup(self):
        """Run the complete setup process"""
        try:
            print("\n=== Starting Creator Setup ===")
            
            # Step 1: Get Channel Info
            print("\1️⃣ Fetching Channel Information...")
            channel_info = self.get_channel_info()
            
            # Step 2: Extract Transcripts
            print("\n2️⃣ Extracting YouTube Transcripts...")
            extractor = TranscriptExtractor(self.youtube_api_key)
            csv_path = extractor.extract_for_channel(self.channel_id)
            
            # Step 3: Create Database
            print("\n3️⃣ Creating Searchable Database...")
            db_creator = DatabaseCreator()
            faiss_path, texts_path = db_creator.create_for_channel(self.channel_id)
            
            # Step 4: Set up Streamlit
            print("\n4️⃣ Setting up Streamlit App...")
            app_dir = self.setup_streamlit(channel_info)
            
            # Step 5: Save Creator Info
            self.save_creator_info(channel_info)
            
            print(f"\n✨ Setup Complete! ✨")
            print(f"Navigate to {app_dir} and run: streamlit run streamlit_app.py")
            
            return app_dir
            
        except Exception as e:
            print(f"\n❌ Error during setup: {str(e)}")
            raise

    def save_creator_info(self, channel_info):
        """Save creator information"""
        info_path = f"{self.creator_dir}/creator_info.json"
        with open(info_path, 'w') as f:
            json.dump(channel_info, f, indent=4)

    def setup_streamlit(self, channel_info):
        """Prepare files for Streamlit deployment"""
        app_dir = f"{self.creator_dir}/streamlit_app"
        os.makedirs(app_dir, exist_ok=True)
        
        # Copy necessary files
        shutil.copy("templates/streamlit_app.py", f"{app_dir}/streamlit_app.py")
        shutil.copy(f"{self.creator_dir}/faiss_index", f"{app_dir}/faiss_index")
        shutil.copy(f"{self.creator_dir}/texts.npy", f"{app_dir}/texts.npy")
        
        # Copy assets
        if os.path.exists(f"{self.creator_dir}/assets"):
            shutil.copytree(
                f"{self.creator_dir}/assets",
                f"{app_dir}/assets",
                dirs_exist_ok=True
            )
        
        # Create .streamlit config directory
        config_dir = os.path.join(app_dir, ".streamlit")
        os.makedirs(config_dir, exist_ok=True)
        
        # Create config.toml with creator's branding
        with open(os.path.join(config_dir, "config.toml"), "w") as f:
            f.write(f"""
[theme]
primaryColor="#FF4B4B"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"
            """)
        
        # Create creator info file
        with open(f"{app_dir}/creator_info.json", 'w') as f:
            json.dump(channel_info, f, indent=4)
        
        # Modify the app file to include the OpenAI API key
        with open(f"{app_dir}/streamlit_app.py", "r") as f:
            content = f.read()
        
        content = content.replace(
            'your-api-key-here',
            self.openai_api_key
        )
        
        with open(f"{app_dir}/streamlit_app.py", "w") as f:
            f.write(content)
            
        return app_dir

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
    
    setup.run_setup()

if __name__ == "__main__":
    main()