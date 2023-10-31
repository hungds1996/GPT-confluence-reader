import os
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from constants import *


class ConfluenceQA:
    def __init__(self, config: dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None

    def init_embeddings(self) -> None:
        self.embedding = OpenAIEmbeddings()

    def init_models(self) -> None:
        self.llm = ChatOpenAI(model=LLM_OPENAI_GPT35, temperature=0.0)

    def vector_db_confluence_docs(self, force_reload: bool = False) -> None:
        """Create a new vector db for the embeddings and persist or load an existing db from directory

        Args:
            force_reload (bool, optional): Flag to force create a new db or not. Defaults to False.
        """

        persist_directory = self.config.get("persist_directory", None)
        confluence_url = self.config.get("confluence_url", None)
        username = self.config.get("username", None)
        api_key = self.config.get("api_key", None)
        space_key = self.config.get("space_key", None)

        if persist_directory and os.path.exists(persist_directory) and not force_reload:
            self.vectordb = Chroma(
                persist_directory=persist_directory, embedding_function=self.embedding
            )
        else:
            # 1.Extract documents from link
            loader = ConfluenceLoader(
                url=confluence_url, username=username, api_key=api_key
            )
            documents = loader.load(space_key=space_key, limit=100, max_pages=10_000)

            # 2. split the texts
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(documents=documents)
            token_splitter = TokenTextSplitter(
                chunk_size=500, chunk_overlap=50, encoding_name="cl100k_base"
            )  # This the encoding for text-embedding-ada-00 model
            texts = token_splitter.split_documents(texts)

            ## 3. Create Embeddings and add to chroma store
            ##TODO: Validate if self.embedding is not None
            self.vectordb = Chroma.from_documents(
                documents=texts,
                embedding=self.embedding,
                persist_directory=persist_directory,
            )

    def retrieval_qa_chain(self):
        """Retrieval chain using vectordb as retriever and LLM to react to prompt"""
        ##TODO: Use custom prompt
        custom_prompt_template = """
            I want you to ANSWER a QUESTION based on the following pieces of CONTEXT. 

            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Your ANSWER should be analytical and straightforward. 
            Try to share deep, thoughtful insights and explain complex ideas in a simple and concise manner. 
            When appropriate use analogies and metaphors to illustrate your point. 
            Your ANSWER should have a strong focus on clarity, logic, and brevity.
            Your ANSWER should be truthful and correct according to the given SOURCES

            CONTEXT: {context}

            QUESTION: {question}

            ANSWER:
        """
        CUSTOMPROMPT = PromptTemplate(
            template=custom_prompt_template, input_variables=["context", "question"]
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 4})
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.retriever
        )
        self.qa.combine_documents_chain.llm_chain.prompt = CUSTOMPROMPT

    def answer_confluence(self, question: str) -> str:
        """To answer question

        Args:
            question (str): input prompt

        Returns:
            str: output answer
        """
        answer = self.qa.run(question)
        return answer
