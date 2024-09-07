import streamlit as st
from utilities import MyGroq
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
my_groq = MyGroq()


if __name__ == "__main__":
    st.title("AI Jarvis V2")

    query = st.input("query")
    if st.button("Submit"):
        res = my_groq(query)
        st.markdown(res)
    


