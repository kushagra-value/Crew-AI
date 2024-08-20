from crewai import Agent
from tools import yt_tool
import faiss
import numpy as np
from streamlit import st

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

# Streamlit UI for API Key and Model Input
st.title("Configuration Settings")

# Prompt for OpenAI API Key
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Prompt for OpenAI Model Name
openai_model_name = st.selectbox(
    "Select your OpenAI Model",
    ["gpt-3.5-turbo"]
)

# Check if the API key is provided
if openai_api_key and openai_model_name:
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
        },
        openai_api_key=openai_api_key,  # Pass API key dynamically
        openai_model_name=openai_model_name  # Pass selected model dynamically
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
        allow_delegation=False,
        openai_api_key=openai_api_key,  # Pass API key dynamically
        openai_model_name=openai_model_name  # Pass selected model dynamically
    )
else:
    st.warning("Please enter your OpenAI API Key and select a model to configure the agents.")
