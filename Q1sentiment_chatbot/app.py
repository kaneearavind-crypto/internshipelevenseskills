import streamlit as st
from chatbot import SentimentChatbot


# -----------------------------------------
# Page Configuration
# -----------------------------------------

st.set_page_config(
    page_title="Sentiment-Aware Chatbot",
    page_icon="🤖",
    layout="centered"
)


# -----------------------------------------
# Application Title
# -----------------------------------------

st.title("🤖 Sentiment-Aware Chatbot")

st.write(
    "This chatbot analyzes the sentiment of your message "
    "and responds appropriately."
)


# -----------------------------------------
# Initialize Chatbot
# -----------------------------------------

@st.cache_resource
def load_chatbot():

    return SentimentChatbot()


chatbot = load_chatbot()


# -----------------------------------------
# Chat History
# -----------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# -----------------------------------------
# Display Previous Messages
# -----------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# -----------------------------------------
# User Input
# -----------------------------------------

user_message = st.chat_input(
    "Type your message here..."
)


if user_message:

    # Add user message to history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    # Display user message
    with st.chat_message("user"):

        st.markdown(user_message)


    # Get chatbot response
    result = chatbot.get_response(user_message)


    # Display chatbot response
    with st.chat_message("assistant"):

        st.markdown(result["response"])

        st.divider()

        st.write(
            f"**Sentiment:** {result['sentiment']}"
        )

        st.write(
            f"**Confidence:** "
            f"{result['confidence'] * 100:.2f}%"
        )


    # Store bot response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                f"{result['response']}\n\n"
                f"**Sentiment:** {result['sentiment']}\n\n"
                f"**Confidence:** "
                f"{result['confidence'] * 100:.2f}%"
            )
        }
    )