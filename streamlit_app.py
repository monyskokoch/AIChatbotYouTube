import streamlit as st
import os
import json
from pathlib import Path

st.set_page_config(
    page_title="Creator AI Chatbots",
    page_icon="🤖",
    layout="wide"
)

# Get list of creators
def get_creator_folders():
    creators_path = Path("creators")
    if creators_path.exists():
        return [f.name for f in creators_path.iterdir() if f.is_dir()]
    return []

# Main interface
st.title("💬 Creator AI Chatbots")

# Sidebar for creator selection
with st.sidebar:
    st.header("Select Creator")
    creators = get_creator_folders()
    
    if not creators:
        st.warning("No creator bots available yet!")
    else:
        selected_creator = st.selectbox(
            "Choose a creator to chat with:",
            creators
        )
        
        # Load creator info if available
        info_path = Path(f"creators/{selected_creator}/creator_info.json")
        if info_path.exists():
            with open(info_path, 'r') as f:
                creator_info = json.load(f)
            st.write(f"**{creator_info['name']}**")
            st.write(creator_info['description'][:200] + "...")
            st.write(f"Subscribers: {int(creator_info['subscriber_count']):,}")

# Main chat area
if 'messages' not in st.session_state:
    st.session_state.messages = []

if creators:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Here we'll add the actual chat logic
                response = "This is a placeholder response. The actual chatbot integration is coming!"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👋 Welcome! No creator bots are set up yet. Use the setup script to add your first creator!")