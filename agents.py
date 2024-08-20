import streamlit as st
from crewai import Agent
from tools import yt_tool
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os

# Ask the user to input the OpenAI API key and model name
openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
openai_model_name = st.text_input("Enter the OpenAI Model Name", value="gpt-4-0125-preview")

# Set the environment variables with the user input
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_MODEL_NAME"] = openai_model_name

    # Function to create vector embeddings using FAISS
    def vector_embedding():
        if "vectors" not in st.session_state:
            st.session_state.embeddings = OpenAIEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
            st.session_state.docs = st.session_state.loader.load()  # Document Loading
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # Splitting
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

    # Create the blog researcher agent
    blog_researcher = Agent(
        role='Blog Researcher from Youtube Videos',
        goal='get the relevant video transcription for the topic {topic} from the provided YT channel',
        verbose=True,
        memory=True,
        backstory=(
           "Expert in understanding videos in AI, Data Science, Machine Learning, and Gen AI, and providing suggestions."
        ),
        tools=[yt_tool],
        allow_delegation=True
    )

    # Create the blog writer agent
    blog_writer = Agent(
        role='Blog Writer',
        goal='Narrate compelling tech stories about the video {topic} from YT video',
        verbose=True,
        memory=True,
        backstory=(
            "With a flair for simplifying complex topics, you craft"
            "engaging narratives that captivate and educate, bringing new"
            "discoveries to light in an accessible manner."
        ),
        tools=[yt_tool],
        allow_delegation=False
    )

    st.title("Nvidia NIM Demo")

    prompt = st.text_input("Enter Your Question From Documents")

    if st.button("Documents Embedding"):
        vector_embedding()
        st.write("Vector Store DB Is Ready")

    if prompt:
        document_chain = create_stuff_documents_chain(blog_writer, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt})
        st.write(f"Response time: {time.process_time() - start} seconds")
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
else:
    st.warning("Please enter your OpenAI API key to proceed.")
