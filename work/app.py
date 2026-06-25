import os
import chromadb
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://host.docker.internal:12345/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-qwen3-embedding-8b")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4-e2b-it")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION = os.getenv("COLLECTION", "kb_realestate_2024")
PDF_PATH = os.getenv("PDF_PATH", "./2024_KB_부동산_보고서_최종.pdf")

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        base_url=LMSTUDIO_BASE_URL,
        api_key=LMSTUDIO_API_KEY,
        check_embedding_ctx_length=False,
    )

@st.cache_resource
def initialize_vectorstore():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    vs = Chroma(client=client, collection_name=COLLECTION, embedding_function=get_embeddings())
    if vs._collection.count() == 0:
        documents = PyPDFLoader(PDF_PATH).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        vs.add_documents(splitter.split_documents(documents))
    return vs

@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

    template = """당신은 KB 부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해부세요.

    컨텍스트: {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])

    model = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LMSTUDIO_BASE_URL,
        api_key=LMSTUDIO_API_KEY,
        temperature=0.7,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )

    store = {}

    def get_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        base_chain,
        get_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="\U0001F3E0")
    st.title("\U0001F3E0 KB 부동산 보고서 AI 어드바이저")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    # 최초 기동 시 1회 초기화 (ChromaDB 연결 + 컬렉션 비어있으면 PDF 임베딩 + 체인 구성)
    # st.cache_resource로 캐시되어 이후 재실행/질문 시 재사용
    with st.spinner("지식베이스 초기화 중... (최초 1회 PDF 임베딩, 시간 소요)"):
        chain = initialize_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("부동산 관련 질문을 입력하세요"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role":"user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain.invoke(
                    {"question": prompt},
                    {"configurable": {"session_id": "streamlit_session"}},
                )
            st.markdown(response)
        st.session_state.messages.append({"role":"assistant", "content": response})

if __name__ == "__main__":
    main()
