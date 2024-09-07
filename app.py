import streamlit as st
import os 

os.environ["GROQ_API_KEY"] == st.secrets["GROQ_API_KEY"]


if __name__ == "__main__":
    st.title("AI Jarvis V2")

