from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import time
from tqdm import tqdm
import os

class TranscriptExtractor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)

    def get_video_ids(self, channel_id, max_videos=50):
        """Fetches video IDs from a YouTube channel with live progress updates."""
        video_ids = []
        next_page_token = None
        print(f"🔍 Fetching videos for channel: {channel_id}")
        
        while len(video_ids) < max_videos:
            request = self.youtube.search().list(
                part="id",
                channelId=channel_id,
                maxResults=min(50, max_videos - len(video_ids)),
                pageToken=next_page_token,
                type="video"
            )
            
            try:
                response = request.execute()
                for item in response["items"]:
                    video_ids.append(item["id"]["videoId"])
                    
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
                    
            except Exception as e:
                print(f"Error fetching videos: {str(e)}")
                break
        
        return video_ids

    def get_transcript(self, video_id):
        """Get transcript for a single video"""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([entry["text"] for entry in transcript])
        except:
            return None

    def extract_for_channel(self, channel_id, output_dir=None, max_videos=50):
        """Extract transcripts for a channel"""
        # Get video IDs
        video_ids = self.get_video_ids(channel_id, max_videos)
        
        # Extract transcripts
        print("📝 Extracting transcripts... This may take a few minutes.")
        transcripts = {}
        
        for video_id in tqdm(video_ids, desc="Processing videos"):
            transcript = self.get_transcript(video_id)
            if transcript:
                transcripts[video_id] = transcript
            time.sleep(0.2)  # Avoid API rate limits
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "transcripts.csv")
        else:
            csv_path = "transcripts.csv"
        
        # Save to CSV
        df = pd.DataFrame(transcripts.items(), columns=["Video_ID", "Transcript"])
        df.to_csv(csv_path, index=False)
        
        print(f"\n✅ Extracted {len(transcripts)} transcripts to {csv_path}")
        return csv_path

if __name__ == "__main__":
    # For testing
    API_KEY = os.getenv('YOUTUBE_API_KEY')
    channel_id = input("Enter YouTube channel ID: ")
    
    extractor = TranscriptExtractor(API_KEY)
    extractor.extract_for_channel(channel_id)