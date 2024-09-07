import streamlit as st
from utilities import MyGroq
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig


GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

class MyGroq:
    def __init__(self):
        pass

    def chat_groq(self, query):
        model = ChatGroq(temperature=0, model_name= "llama-3.1-8b-instant")   # "llama-3.1-8b-instant","llama-guard-3-8b"
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You're a very knowledgeable Machine Learning Engineer.",
                ),
                ("human", "{question}"),
            ]
        )
        runnable = prompt | model | StrOutputParser()
        res = runnable.invoke(query)
        return res
    

if __name__ == "__main__":
    st.title("AI Jarvis V2")
    my_groq = MyGroq()

    query = st.text_input("query")
    if st.button("Submit"):
        res = my_groq(query)
        st.markdown(res)
    


