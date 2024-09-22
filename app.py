### [시작] Delploy 할때만 실행되는 코드 #####################
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
### [종료 ] Delploy 할때만 실행되는 코드 #####################

###  pysqlite3-binary ---> requirements.txt 에 추가
import os
import random
import base64
import numpy as np
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from groq import Groq

### Layout ------------------------------------------------------------------------------------
if "center" not in st.session_state:
    layout = "centered"
else:
    layout = "wide" if st.session_state.center else "centered"
st.set_page_config(page_title="AI Jarvis-v1", layout=layout)

st.markdown(
            """
        <style>
            .st-emotion-cache-1c7y2kd {
                flex-direction: row-reverse;
                text-align: right;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )


@st.fragment
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def apply_bg_image(main_bg, sidebar_bg):
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{main_bg}");
    background-size: 100%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}

    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{sidebar_bg}");
    background-size: 100%;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

parent_dir = Path(__file__).parent
main_image_path = str(parent_dir) +"/main_images"
sidebar_image_path = str(parent_dir) +"/sidebar_images"
sidebar_image_list = [os.path.join(sidebar_image_path,f) for f in os.listdir(sidebar_image_path) if os.path.isfile(os.path.join(sidebar_image_path, f))]
sidebar_img = random.choice(sidebar_image_list)
main_image_list = [os.path.join(main_image_path,f) for f in os.listdir(main_image_path) if os.path.isfile(os.path.join(main_image_path, f))]
main_img = random.choice(main_image_list)
sidebar_bg = get_img_as_base64(f"{sidebar_img}")
main_bg = get_img_as_base64(f"{main_img}")

### API KEY ------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
pw = st.secrets["password"]

### Function --------------------------------------------------------------------------

def calculate_time_delta(start, end):
    delta = end - start
    return delta.total_seconds()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def image_chat(image_path, text, model_name):
    base64_image = encode_image(image_path)
    client = Groq()
    chat_completion  = client.chat.completions.create(
        model=model_name,
        messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

def open_chat(query, model_name):
    model = ChatGroq(temperature=0, model_name= model_name)   # "llama-3.1-8b-instant","llama-guard-3-8b", "gemma2-9b-it","llama-3.1-70b-versatile",
    prompt = ChatPromptTemplate.from_messages([
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

def make_retriever_from_text(context):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(context)
    embeddings_model = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(docs, embeddings_model, persist_directory="./chroma_db_text")
    retriever1 = vectorstore.as_retriever()
    return retriever1

def make_retriever_from_pdf(pdf):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pdf)
    embeddings_model = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings_model, persist_directory="./chroma_db_pdf")
    retriever1 = vectorstore.as_retriever()
    return retriever1

def make_retriever_from_url(url):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(url)
    embeddings_model = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings_model, persist_directory="./chroma_db_url")
    retriever1 = vectorstore.as_retriever()
    return retriever1

def quick_rag_chat(query, retriever, model_name, json_style:bool):

    if json_style:
        system_prompt = ('''
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.

    {context}
    Please provide your answer in the following JSON format: 
    {{
    "answer": "Your detailed answer here",\n
    "keywords: [list of important keywords from the context] \n
    "sources": "Direct sentences or paragraphs from the context that support your answers. ONLY RELEVANT TEXT DIRECTLY FROM THE DOCUMENTS. DO NOT ADD ANYTHING EXTRA. DO NOT INVENT ANYTHING."
    }}
    The JSON must be a valid json format and can be read with json.loads() in Python. Answer:
                        ''')
    
    else: 
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
model_name_dict = {"Llama3.1(8B)":"llama-3.1-8b-instant", 
                   "Gemma2(9B)":"gemma2-9b-it", 
                   "Llama3.1(70B)":"llama-3.1-70b-versatile",
                   "Llava_v1.5(7B)":"llava-v1.5-7b-4096-preview"}

if "login_status" not in st.session_state: st.session_state.login_status = False
if "json_style" not in st.session_state: st.session_state.json_style = True
if "time_delta" not in st.session_state: st.session_state.time_delta = ""
if "rag_time_delta" not in st.session_state: st.session_state.rag_time_delta = ""

if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
if "rag_messages_text" not in st.session_state: st.session_state.rag_messages_text = [{"role": "assistant", "content": "How can I help you?"}]
if "rag_messages_pdf" not in st.session_state: st.session_state.rag_messages_pdf = [{"role": "assistant", "content": "How can I help you?"}]
if "rag_messages_url" not in st.session_state: st.session_state.rag_messages_url = [{"role": "assistant", "content": "How can I help you?"}]

if "chat_history" not in st.session_state: st.session_state.chat_history = ""
if "model_name" not in st.session_state: st.session_state.model_name = ""
if "file_path" not in st.session_state: st.session_state.file_path = ""

if "retriever_text" not in st.session_state: st.session_state.retriever_text = ""
if "retriever_pdf" not in st.session_state: st.session_state.retriever_pdf = ""
if "retriever_url" not in st.session_state: st.session_state.retriever_url = ""

if "retrieval_docs_text" not in st.session_state: st.session_state.retrieval_docs_text = ""
if "retrieval_docs_pdf" not in st.session_state: st.session_state.retrieval_docs_pdf = ""
if "retrieval_docs_url" not in st.session_state: st.session_state.retrieval_docs_url = ""

if "prev_questions" not in st.session_state: st.session_state.prev_questions = []
if "img_answer" not in st.session_state: st.session_state.img_answer = ""

example_text = '''
example text:
The U.S. Commerce Department’s Bureau of Industry and Security (BIS) released a new export control guideline on advanced technologies related to quantum computing and chip manufacturing on Thursday, a measure designed to safeguard national security while restricting technological availability to China.
The BIS published the rule regarding regulations on export items under four categories: quantum computing; advanced chip manufacturing equipment; gate all-around field-effect transistor (Gaafet) technology, which produces or develops high-performance computing chips used in supercomputers; and additive manufacturing items, which are equipment and materials used to produce metal or metal alloy components.
'''


if __name__ == "__main__":
    
    ### Sidebar -----------------------------------------------------------------------------------------------------
    with st.sidebar:
        
        st.title("⚓ :gray[Menu]")
        st.subheader("AI Assistant for you")
        st.markdown("")

        password = st.text_input("🔑 **Password**", type="password")
        col11, col12 = st.columns(2)
        with col11: btn_login=st.button("Login", use_container_width=True, help="1234")
        with col12: btn_init=st.button("Memory Init", use_container_width=True, help="Initialize Multi-Turn Memory")

        if btn_login and password == pw:
            st.session_state.login_status = True
            st.session_state.chat_history = ""
            st.info("Login Success")
        else: pass 

        if btn_init:
            st.session_state.chat_history = ""
            st.session_state.prev_questions = []
        st.markdown("---")
        service_type = st.radio("🐬 **Services**", options=["Open Chat", "Text Rag", "PDF Rag", "URL Rag", "Image Rag"])
        
        st.markdown("---")

        if service_type == "Image Rag":
            llm1 = st.radio("🐬 **Select LLM**", options=["Llava_v1.5(7B)"], index=0, key="dsfv", help="Using Groq API")
        else: 
            llm1 = st.radio("🐬 **Select LLM**", options=["Llama3.1(8B)", "Gemma2(9B)", "Llama3.1(70B)"], index=0, key="dsfv", help="Using Groq API")
            st.markdown("")
            st.session_state.json_style = st.checkbox("Json Type Rag Response", value=True)
        st.session_state.model_name = model_name_dict[llm1]
        st.warning(st.session_state.model_name)
        st.markdown("---")

    ## Main -----------------------------------------------------------------------------------------------
    st.title("🧭 :blue[AI Jarvis v1]")
    col31, col32 = st.columns(2)
    with col31: st.checkbox("🐋 Wide Layout", key="center", value=st.session_state.get("center", False))
    with col32: bg_check = st.checkbox("background Image", value=True)
    if bg_check: apply_bg_image(main_bg, sidebar_bg)
    else: pass
    st.markdown("---")
    
    if service_type == "Open Chat" and st.session_state.login_status:
        text_input1 = st.chat_input(placeholder="Say something")
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
            st.success(f"⏱️ Latency(Sec) : {np.round(st.session_state.time_delta,2)}  /  Total Q&A Length(Char): {len(st.session_state.chat_history)}")

        try:
            if text_input1 and text_input1 not in st.session_state.prev_questions:
                st.session_state.prev_questions.append(text_input1)
                # selected = pills("Previous Questions", st.session_state.prev_questions)
            else: 
                pass
                # selected = pills("Previous Questions", st.session_state.prev_questions)
        except: pass

    elif service_type == "Text Rag" and st.session_state.login_status:
        with st.expander("Quick Reference Texts", expanded=True): 
            context_input = st.text_area("", example_text, key="uyhv", height=200)
            with st.spinner("Processing.."):
                if st.button("Create Retriever"):
                    try:
                        vectordb = Chroma(persist_directory="chroma_db_text", embedding_function=OpenAIEmbeddings())
                        vectordb._client.delete_collection(vectordb._collection.name)
                    except: pass
                    st.session_state.retriever_text  = make_retriever_from_text(context_input)

            if st.session_state.retriever_text:
                st.info(st.session_state.retriever_text)

        text_input2 = st.chat_input("Say something")
        if text_input2:
            st.session_state.chat_history = st.session_state.chat_history + "\n" + text_input2 + "\n"
            rag_start_time = datetime.now()
            st.session_state.rag_messages_text.append({"role": "user", "content": text_input2})
            retrieval_docs, output2 = quick_rag_chat(text_input2, st.session_state.retriever_text, st.session_state.model_name, st.session_state.json_style)
            st.session_state.rag_messages_text.append({"role": "assistant", "content": output2})
            st.session_state.retrieval_docs_text = retrieval_docs
            rag_end_time = datetime.now()
            st.session_state.rag_time_delta = calculate_time_delta(rag_start_time, rag_end_time)
            st.session_state.chat_history = st.session_state.chat_history + "\n" + output2 + "\n"
            
        for msg in st.session_state.rag_messages_text:
            if msg["role"] == "user":
                st.chat_message(msg["role"], avatar="🐬").write(msg["content"])
            else:
                st.chat_message(msg["role"], avatar="🤖").write(msg["content"])
                
        with st.expander("Retrieval Docs"): st.session_state.retrieval_docs_text

        if st.session_state.rag_time_delta: 
            st.success(f"⏱️ Latency(Sec) : {np.round(st.session_state.rag_time_delta,2)}  /  Total Q&A Length(Char): {len(st.session_state.chat_history)}")

        try:
            if text_input2 and text_input2 not in st.session_state.prev_questions:
                st.session_state.prev_questions.append(text_input2)
            else: 
                pass
        except: pass
    
    elif service_type == "PDF Rag" and st.session_state.login_status:
        with st.expander("📎:green[**Upload Your PDF**]", expanded=True):
            parent_dir = Path(__file__).parent
            base_dir = str(parent_dir) + "\data"
            uploaded_file = st.file_uploader("", type=['PDF', 'pdf'])
            btn1 = st.button("Create PDF Retreiver", type='secondary')
            try:
                with st.spinner("processing..."):
                    if uploaded_file and btn1:
                        st.session_state.retrieval_docs_pdf = ""
                        if not os.path.exists(base_dir):
                            os.makedirs(base_dir)

                        files = os.listdir(base_dir)
                        for file in files:
                            file_path = os.path.join(base_dir, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)

                        temp_dir = base_dir 
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        try:
                            vectordb = Chroma(persist_directory="chroma_db_pdf", embedding_function=OpenAIEmbeddings())
                            vectordb._client.delete_collection(vectordb._collection.name)
                        except: pass
                        st.session_state.retriever_pdf = make_retriever_from_pdf(docs)
                    st.info(st.session_state.retriever_pdf)
            except: st.warning("There are some errors in your PDF")

        text_input3 = st.chat_input("Say something")
        if text_input3:
            st.session_state.chat_history = st.session_state.chat_history + "\n" + text_input3 + "\n"
            rag_start_time = datetime.now()
            st.session_state.rag_messages_pdf.append({"role": "user", "content": text_input3})
            retrieval_docs3, output3 = quick_rag_chat(text_input3, st.session_state.retriever_pdf, st.session_state.model_name, st.session_state.json_style)
            st.session_state.rag_messages_pdf.append({"role": "assistant", "content": output3})
            st.session_state.retrieval_docs_pdf = retrieval_docs3
            rag_end_time = datetime.now()
            st.session_state.rag_time_delta = calculate_time_delta(rag_start_time, rag_end_time)
            st.session_state.chat_history = st.session_state.chat_history + "\n" + output3 + "\n"
            
        for msg in st.session_state.rag_messages_pdf:
            if msg["role"] == "user":
                st.chat_message(msg["role"], avatar="🐬").write(msg["content"])
            else:
                st.chat_message(msg["role"], avatar="🤖").write(msg["content"])
                
        with st.expander("Retrieval Docs"): st.session_state.retrieval_docs_pdf

        if st.session_state.rag_time_delta: 
            st.success(f"⏱️ Latency(Sec) : {np.round(st.session_state.rag_time_delta,2)}  /  Total Q&A Length(Char): {len(st.session_state.chat_history)}")

    elif service_type == "URL Rag" and st.session_state.login_status:
        url = st.text_input("🕸️ :green[**URL**]")
        btn3 = st.button("Create Retriever3")
        if url and btn3:
            loader = WebBaseLoader(url)
            docs = loader.load()
            try:
                vectordb = Chroma(persist_directory="chroma_db_url", embedding_function=OpenAIEmbeddings())
                vectordb._client.delete_collection(vectordb._collection.name)
            except: pass
            st.session_state.retriever_url = make_retriever_from_url(docs)
        st.info(st.session_state.retriever_url)


        text_input4 = st.chat_input("Say something")
        if text_input4:
            st.session_state.chat_history = st.session_state.chat_history + "\n" + text_input4 + "\n"
            rag_start_time = datetime.now()
            st.session_state.rag_messages_url.append({"role": "user", "content": text_input4})
            retrieval_docs4, output4 = quick_rag_chat(text_input4, st.session_state.retriever_url, st.session_state.model_name, st.session_state.json_style)
            st.session_state.rag_messages_url.append({"role": "assistant", "content": output4})
            st.session_state.retrieval_docs_url = retrieval_docs4
            rag_end_time = datetime.now()
            st.session_state.rag_time_delta = calculate_time_delta(rag_start_time, rag_end_time)
            st.session_state.chat_history = st.session_state.chat_history + "\n" + output4 + "\n"
            
        for msg in st.session_state.rag_messages_url:
            if msg["role"] == "user":
                st.chat_message(msg["role"], avatar="🐬").write(msg["content"])
            else:
                st.chat_message(msg["role"], avatar="🤖").write(msg["content"])
                
        with st.expander("Retrieval Docs"): st.session_state.retrieval_docs_url

        if st.session_state.rag_time_delta: 
            st.success(f"⏱️ Latency(Sec) : {np.round(st.session_state.rag_time_delta,2)}  /  Total Q&A Length(Char): {len(st.session_state.chat_history)}")

        pass 

    elif service_type == "Image Rag" and st.session_state.login_status:
        with st.expander("📎:green[**Upload Your Image**]", expanded=True):
            parent_dir = Path(__file__).parent
            base_dir = str(parent_dir) + "\image"
            uploaded_file = st.file_uploader("", type=['jpg', 'png'])
            btn2 = st.button("Save", type='secondary')
            with st.spinner("processing..."):
                if uploaded_file and btn2:
                    if not os.path.exists(base_dir):
                        os.makedirs(base_dir)
                    files = os.listdir(base_dir)
                    for file in files:
                        file_path = os.path.join(base_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    temp_dir = base_dir 
                    st.session_state.file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(st.session_state.file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                    st.markdown(st.session_state.file_path)
        try: st.image(st.session_state.file_path, width=600)
        except: pass

        text1 = st.text_input("Input your Query")
        if st.button("Asking"):
            st.session_state.img_answer = image_chat(st.session_state.file_path, text1, st.session_state.model_name)
        
        st.info(st.session_state.img_answer)

    else: pass
    