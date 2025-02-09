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

    def extract_for_channel(self, channel_id, max_videos=50):
        """Main function to extract transcripts for a channel"""
        # Create creator directory
        creator_dir = f"creators/{channel_id}"
        os.makedirs(creator_dir, exist_ok=True)
        
        # Get video IDs
        print(f"🔍 Fetching videos for channel: {channel_id}")
        video_ids = self._get_video_ids(channel_id, max_videos)
        
        # Get transcripts
        transcripts = {}
        print("📝 Extracting transcripts... This may take a few minutes.")
        
        for video_id in tqdm(video_ids, desc="Processing videos"):
            transcript = self._get_transcript(video_id)
            if transcript:
                transcripts[video_id] = transcript
            time.sleep(0.2)  # Avoid API rate limits

        # Save to CSV in creator's directory
        df = pd.DataFrame(transcripts.items(), columns=["Video_ID", "Transcript"])
        csv_path = f"{creator_dir}/transcripts.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n✅ Extracted {len(transcripts)} transcripts to {csv_path}")
        return csv_path

    def _get_video_ids(self, channel_id, max_videos):
        """Get video IDs from a channel"""
        video_ids = []
        next_page_token = None
        
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

    def _get_transcript(self, video_id):
        """Get transcript for a single video"""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([entry["text"] for entry in transcript])
        except:
            return None

# Only run this if script is run directly
if __name__ == "__main__":
    API_KEY = os.getenv('YOUTUBE_API_KEY')  # Replace with your API key
    channel_id = input("Enter YouTube channel ID: ")
    
    extractor = TranscriptExtractor(API_KEY)
    extractor.extract_for_channel(channel_id)