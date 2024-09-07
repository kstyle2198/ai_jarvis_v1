import os
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

st.set_page_config(page_title="AI Jarvis-v1", layout="wide")
st.markdown(
            """
        <style>
            .st-emotion-cache-janbn0 {
                flex-direction: row-reverse;
                text-align: right;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
pw = st.secrets["password"]

def calculate_time_delta(start, end):
    delta = end - start
    return delta.total_seconds()

def open_chat(query, model_name):
    model = ChatGroq(temperature=0, model_name= model_name)   # "llama-3.1-8b-instant","llama-guard-3-8b", "gemma2-9b-it","llama-3.1-70b-versatile",
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
    

model_name_dict = {"Llama3.1(8B)":"llama-3.1-8b-instant", "Gemma2(9B)":"gemma2-9b-it", "Llama3.1(70B)":"llama-3.1-70b-versatile"}

if "password" not in st.session_state:
    st.session_state.password = ""

if "time_delta" not in st.session_state:
    st.session_state.time_delta = ""
    st.session_state.rag_time_delta = ""
    
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.reversed_messages = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""


if __name__ == "__main__":
    ### Sidebar -----------------------------------------------------------------------------------------------------
    with st.sidebar:
        st.title("⚓ AI Jarvis v1")
        st.markdown("")
        password = st.text_input("🔑 Password", type="password")
        if st.button("Login") and password == pw:
            st.session_state.password = password
        else: st.warning("Check your password")

        st.markdown("---")
        service_type = st.radio("🐬 Services", options=["Open Chat", "Rag", ])
    
        st.markdown("---")
        if service_type == "Open Chat":
            llm1 = st.radio("🐬 **Select LLM**", options=["Llama3.1(8B)", "Gemma2(9B)", "Llama3.1(70B)"], index=0, key="dsfv", help="Bigger LLM returns better answers but takes more time")
            model_name = model_name_dict[llm1]
            model_name
            st.markdown("")



        

    ## Main -----------------------------------------------------------------------------------------------
    st.title("AI Jarvis v1")
    st.markdown("---")

    try:

        if service_type == "Open Chat" and st.session_state.password:
            text_input1 = st.chat_input("Say something")
            st.session_state.chat_history = st.session_state.chat_history + "\n" + text_input1 + "\n"

            if text_input1:
                start_time = datetime.now()
                st.session_state.messages.append({"role": "user", "content": text_input1})
                output1 = open_chat(st.session_state.chat_history, model_name)
                st.session_state.messages.append({"role": "assistant", "content": output1})
                end_time = datetime.now()
                st.session_state.time_delta = calculate_time_delta(start_time, end_time)

                st.session_state.chat_history = st.session_state.chat_history + "\n" + output1 + "\n"
                
                
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
                else:
                    st.chat_message(msg["role"], avatar="🤖").write(msg["content"])
            if st.session_state.time_delta: 
                st.success(f"⏱️ Latency(Sec) : {st.session_state.time_delta}")
            
        else: pass
    except:
        pass




