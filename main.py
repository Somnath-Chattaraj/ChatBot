import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = "RAG"

os.environ["GOOGLE_API_KEY"] = ""

import warnings

warnings.filterwarnings("ignore")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)
print(model.invoke("what is the capital of india?").content)

import bs4
from langchain import hub

from langchain.chains import create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_chroma import Chroma

from langchain_community.document_loaders import WebBaseLoader

from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import MessagesPlaceholder

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
)

doc = loader.load()

doc

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

splits = text_splitter.split_documents(doc)

splits

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=gemini_embeddings,
)

retriever = vectorstore.as_retriever()

retriever

system_prompt = (
    "You are an assistant for question answering tasks. "
    "Always give a correct answer"
    "Use the following pieces of retrieved context to answer the question "
    "If you don't know the answer, say that you don't know. And dont give any false answer"
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answering_chain = create_stuff_documents_chain(model, chat_prompt)

rag_chain = create_retrieval_chain(retriever, question_answering_chain)

rag_chain.invoke({"input": "Give an relationship advice"})['answer']

from langchain_core.messages import HumanMessage, AIMessage

def start_rag_chat(rag_chain):
    print("💬 RAG Chatbot is now running! Type 'exit' to stop.\n")

    chat_history = []

    while True:
        user_input = input("You: ")

        if user_input.lower() in {"exit", "quit"}:
            print("👋 Exiting chat. Goodbye!")
            break

        # Add user input to history
        chat_history.append(HumanMessage(content=user_input))

        # Invoke the RAG chain with input and history
        try:
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })

            answer = response['answer']
            print("Bot:", answer)

            # Add bot response to history
            chat_history.append(AIMessage(content=answer))

        except Exception as e:
            print("⚠️ Error during RAG invocation:", e)

start_rag_chat(rag_chain)

