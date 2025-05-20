import streamlit as st
from rag_engine import FitnessRAG
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Fitness Coach AI",
    page_icon="💪",
    layout="wide"       
)

# Initialize RAG engine
@st.cache_resource
def init_rag():
    return FitnessRAG()

rag = init_rag()

# Streamlit UI
st.title("🏋️‍♂️ Your AI Fitness Coach")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI Fitness Coach. I can help you with workout plans, nutrition advice, and answer any fitness-related questions. What would you like to know?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anythinwg about fitness!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag.get_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)

# Sidebar with user profile
with st.sidebar:
    st.header("Your Profile")
    fitness_level = st.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced"])
    goals = st.multiselect("Fitness Goals", 
                          ["Weight Loss", "Muscle Gain", "Endurance", "Flexibility", 
                           "General Fitness", "Sports Performance"])
    
    if st.button("Update Profile"):
        st.success("Profile updated successfully!") 