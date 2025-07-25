import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000/chat/"

st.title("Hanan Elhosary Chatbot")

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is on your mind?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            full_response = ""
            try:
                payload = {"user_id": st.session_state.user_id, "message": prompt}
                
                with requests.post(API_URL, json=payload, stream=False) as r:
                    r.raise_for_status()
                    response_data = r.json()
                    bot_response = response_data.get("response", "No response found.")
                    full_response += bot_response
                    message_placeholder.markdown(full_response)

            except requests.exceptions.RequestException as e:
                error_message = f"Error communicating with the API. Please ensure the backend is running. Details: {e}"
                message_placeholder.error(error_message)
                full_response = error_message

    st.session_state.messages.append({"role": "assistant", "content": full_response})