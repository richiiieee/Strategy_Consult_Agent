from langchain_core.tools import BaseTool
from langchain.chains import RetrievalQA
from pydantic import BaseModel
from crewai.tools import tool
from typing import Any,Type
import glob
import logging
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import pdfplumber
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain.chains import RetrievalQA
import os
from langchain.schema import Document as lang_doc
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from crewai.tools import tool
from typing import Any
from flow_functions import _extract_text_from_file
from flow_functions import groq_llm

#name="QueryIndex",description="Retrieves answers to queries using a vector store and LLM."
#vectorstore: Any, query: str, llm: Any

@tool
def parse_and_index_tool(folder_path: dict):
        """This tool processes text files from a folder, splits the text into chunks, generates embeddings for the chunks, 
        and stores them in a FAISS vector store for efficient retrieval.

        Args:
            folder_path (str): Path to the folder containing files to process.

        Returns:
            FAISS: A FAISS vector store containing the indexed document embeddings."""
        all_texts = []
        all_metadata = []

        # Parse files
        for root, _, files in os.walk(folder_path['value']):

            for file in files:

                file_path = os.path.join(root, file)

                text =_extract_text_from_file(file_path)

                if text:
                    all_texts.append(text)
                    all_metadata.append({"source": file})

        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents(all_texts, all_metadata)

        print(docs)

        # Embedding
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    
        document_embeddings = embeddings.embed_documents([doc.page_content for doc in docs])


        embedding_dim = len(document_embeddings[0])

        #creating a vector store
        global vectorstore
        index = faiss.IndexFlatL2(embedding_dim)
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        
        for doc, emb in zip(docs, embeddings):
            vectorstore.add_documents([doc], embeddings=[emb])

        print("vectorstr : ",vectorstore)

        return vectorstore

@tool
def QueryIndex(vectorstore: Any, query: dict ) -> dict:
    """
    Executes a retrieval-based question-answering (QA) task using a vector store and a query.

    Args:
        vectorstore (Any): A vector store object that contains the indexed data and provides retrieval capabilities.
        query (str): The question or query to be answered based on the data in the vector store.
        llm (Any): The language model used for retrieval-based QA.

    Returns:
        dict: A response dictionary containing the query and the generated answer.
    """
    if not groq_llm:
        print("error: LLM is not properly initialized.")
        return {"error": "LLM is not properly initialized."}
    try:
        print("Inside QueryIndex tool...")
        print("VectorSTore : ",vectorstore)
        # Create retriever from the vector store
        retriever = vectorstore.as_retriever()

        # Create a retrieval-based QA chain
        qa_chain = RetrievalQA.from_chain_type(groq_llm , retriever=retriever)

        # Run the QA chain with the query
        response = qa_chain.run({"query": query['value']})

        return {"query": query['value'], "answer": response}

    except Exception as e:
        return {"error": str(e), "message": "Failed to execute QueryIndex tool."}


# ---------------- Monitoring ----------------------------------

# class MyHandler(FileSystemEventHandler):
#     def __init__(self):
#         self.files = []


#     def on_created(self, event):
#         print(f'Event type: {event.event_type}  path: {event.src_path}')
#         # if event.is_directory:
#         #     return
#         self.files.append(event.src_path)

#     def on_modified(self, event):
#         print(f'Event type: {event.event_type}  path: {event.src_path}')
#         self.files.append(event.src_path)

#     def on_deleted(self, event):
#         print(f'Event type: {event.event_type}  path: {event.src_path}')
#         if event.src_path in self.files:
#             self.files.remove(event.src_path)


# @tool
# def monitor_directory_tool(vectorstore):
#     """
#     Tool for monitoring a directory for file system changes.
    
#     :param input_params: A dictionary containing 'path' and 'recursive' keys.
#     :return: A status message after the monitor stops.
#     """
#     path = r"C:\D Drive\Richa\NuMindsAI\Projects\CREWAIPROJECT\strategy_consult_agent\new_dir"
#     observer = Observer()
#     handler = MyHandler()
#     updated_files = handler.files
#     observer.schedule(handler, path, recursive=True)
#     observer.start()

#     print(f"Monitoring started on: {path}")
    

#     try:
#         while True:
#             pass
#     except KeyboardInterrupt:
#         print(f"Updates files list:\n {updated_files}")
#         observer.stop()

#     observer.join()





# ---------------------------------
# print("\nSource Documents:")
# for doc in response['source_documents']:
#     print(f"- Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}...")