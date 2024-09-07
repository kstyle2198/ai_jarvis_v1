import os
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st


from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

###  pysqlite3-binary ---> requirements.txt 에 추가

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

### API KEY ------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
pw = st.secrets["password"]

### Function --------------------------------------------------------------------------
def calculate_time_delta(start, end):
    delta = end - start
    return delta.total_seconds()

def open_chat(query, model_name):
    model = ChatGroq(temperature=0, model_name= model_name)   # "llama-3.1-8b-instant","llama-guard-3-8b", "gemma2-9b-it","llama-3.1-70b-versatile",
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable Shipbuilding Engineer.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    res = runnable.invoke(query)
    return res

def make_retriever(context):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(context)
    embeddings_model = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(docs, embeddings_model)
    retriever = vectorstore.as_retriever()
    return retriever

def rag_chat(query, retriever, model_name):
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    model = ChatGroq(temperature=0, model_name= model_name)
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": f"{query}"})
    return response["context"], response["answer"]





    
### Variables ---------------------------------------------------------------------------------------------
model_name_dict = {"Llama3.1(8B)":"llama-3.1-8b-instant", "Gemma2(9B)":"gemma2-9b-it", "Llama3.1(70B)":"llama-3.1-70b-versatile"}

if "password" not in st.session_state: st.session_state.password = ""

if "time_delta" not in st.session_state:
    st.session_state.time_delta = ""
    st.session_state.rag_time_delta = ""
    
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.reversed_messages = ""

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.reversed_rag_messages = ""

if "chat_history" not in st.session_state: st.session_state.chat_history = ""
if "model_name" not in st.session_state: st.session_state.model_name = ""
if "retriever" not in st.session_state: st.session_state.retriever = ""
if "retrieval_docs" not in st.session_state: st.session_state.retrieval_docs = ""


example_text = '''
The U.S. Commerce Department’s Bureau of Industry and Security (BIS) released a new export control guideline on advanced technologies related to quantum computing and chip manufacturing on Thursday, a measure designed to safeguard national security while restricting technological availability to China.
The BIS published the rule regarding regulations on export items under four categories: quantum computing; advanced chip manufacturing equipment; gate all-around field-effect transistor (Gaafet) technology, which produces or develops high-performance computing chips used in supercomputers; and additive manufacturing items, which are equipment and materials used to produce metal or metal alloy components.
'''


if __name__ == "__main__":
    ### Sidebar -----------------------------------------------------------------------------------------------------
    with st.sidebar:
        st.title("⚓ AI Jarvis v1")
        st.markdown("")
        password = st.text_input("🔑 Password", type="password")
        
        col11, col12 = st.columns(2)
        with col11: btn_login=st.button("Login", use_container_width=True)
        with col12: btn_init=st.button("Memory Init", use_container_width=True)

        if btn_login and password == pw:
            st.session_state.password = password
            st.info("Login Success")
        else: pass 

        if btn_init:st.session_state.chat_history = ""

        st.markdown("---")
        service_type = st.radio("🐬 Services", options=["Open Chat", "Rag Chat",])
    
        st.markdown("---")
        if service_type == "Open Chat":
            llm1 = st.radio("🐬 **Select LLM**", options=["Llama3.1(8B)", "Gemma2(9B)", "Llama3.1(70B)"], index=0, key="dsfv", help="Bigger LLM returns better answers but takes more time")
            st.session_state.model_name = model_name_dict[llm1]
            st.markdown("")
        elif service_type == "Rag Chat":
            llm2 = st.radio("🐬 **Select LLM**", options=["Llama3.1(8B)", "Gemma2(9B)", "Llama3.1(70B)"], index=0, key="dsfv", help="Bigger LLM returns better answers but takes more time")
            st.session_state.model_name = model_name_dict[llm2]
            st.markdown("")
        

    ## Main -----------------------------------------------------------------------------------------------
    st.title("AI Jarvis v1")
    st.markdown("---")


    if service_type == "Open Chat" and st.session_state.password:
        text_input1 = st.chat_input("Say something")
        if text_input1:
            st.session_state.chat_history = st.session_state.chat_history + "\n" + text_input1 + "\n"
            start_time = datetime.now()
            st.session_state.messages.append({"role": "user", "content": text_input1})
            output1 = open_chat(st.session_state.chat_history, st.session_state.model_name)
            st.session_state.messages.append({"role": "assistant", "content": output1})
            end_time = datetime.now()
            st.session_state.time_delta = calculate_time_delta(start_time, end_time)
            st.session_state.chat_history = st.session_state.chat_history + "\n" + output1 + "\n"

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message(msg["role"], avatar="🐬").write(msg["content"])
            else:
                st.chat_message(msg["role"], avatar="🤖").write(msg["content"])
        if st.session_state.time_delta: 
            st.success(f"⏱️ Latency(Sec) : {st.session_state.time_delta} / Input Char Length: {len(st.session_state.chat_history)}")

    elif service_type == "Rag Chat" and st.session_state.password:
        with st.expander("VectorStore"): 
            context_input = st.text_area("Reference Knowledge",example_text, height=200)
            with st.spinner("Processing.."):
                if st.button("Create Retriever"):
                    st.session_state.retriever = ""
                    st.session_state.retriever = make_retriever(context_input)
                    st.info("Retriever is created")
            if st.session_state.retriever:
                st.session_state.retriever

        text_input2 = st.chat_input("Say something")
        if text_input2:
            st.session_state.chat_history = st.session_state.chat_history + "\n" + text_input2 + "\n"
            start_time = datetime.now()
            st.session_state.rag_messages.append({"role": "user", "content": text_input2})

            retrieval_docs, output2 = rag_chat(text_input2, st.session_state.retriever, st.session_state.model_name)

            st.session_state.rag_messages.append({"role": "assistant", "content": output2})
            st.session_state.retrieval_docs = retrieval_docs
            end_time = datetime.now()
            st.session_state.time_delta = calculate_time_delta(start_time, end_time)
            st.session_state.chat_history = st.session_state.chat_history + "\n" + output2 + "\n"
        
        for msg in st.session_state.rag_messages:
            if msg["role"] == "user":
                st.chat_message(msg["role"], avatar="🐬").write(msg["content"])
            else:
                st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

        with st.expander("Retrieval Docs"): st.session_state.retrieval_docs

        if st.session_state.time_delta: 
            st.success(f"⏱️ Latency(Sec) : {st.session_state.time_delta}/ Input Char Length: {len(st.session_state.chat_history)}")



    else: pass



    



