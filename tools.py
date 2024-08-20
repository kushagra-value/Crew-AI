# tool.py
from crewai_tools import YoutubeChannelSearchTool

# Initialize the tool with a specific Youtube channel handle to target your search
yt_tool = YoutubeChannelSearchTool(youtube_channel_handle='@krishnaik06')

# Update this code based on your needs if ChromaDB is not required
# or find an alternative that fits your use case
