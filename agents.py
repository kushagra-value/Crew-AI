from crewai import Agent
from tools import yt_tool
import faiss                   # Ensure FAISS is installed
import numpy as np
from dotenv import load_dotenv

load_dotenv()

import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4-0125-preview"

# Example FAISS setup
class FAISSIndex:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_map = {}

    def add_vectors(self, vectors, ids):
        self.index.add(vectors)
        for i, id in enumerate(ids):
            self.id_map[id] = i

    def search(self, query_vector, k=5):
        distances, indices = self.index.search(query_vector, k)
        return [(self.id_map.get(idx, idx), distance) for idx, distance in zip(indices[0], distances[0])]

# Create a FAISS index
faiss_index = FAISSIndex(dimension=768)  # Adjust dimension according to your embeddings

# Create a senior blog content researcher
blog_researcher = Agent(
    role='Blog Researcher from Youtube Videos',
    goal='Get the relevant video transcription for the topic {topic} from the provided YT channel and index it using FAISS',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI Data Science, Machine Learning, and GEN AI and providing suggestions. Utilizes FAISS for efficient retrieval."
    ),
    tools=[yt_tool],
    allow_delegation=True,
    custom_methods={
        'index_content': lambda content, ids: faiss_index.add_vectors(content, ids),
        'search_content': lambda query_vector, k: faiss_index.search(query_vector, k)
    }
)

# Creating a senior blog writer agent with YT tool
blog_writer = Agent(
    role='Blog Writer',
    goal='Narrate compelling tech stories about the video {topic} from YT video',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft engaging narratives that captivate and educate, bringing new discoveries to light in an accessible manner."
    ),
    tools=[yt_tool],
    allow_delegation=False
)
