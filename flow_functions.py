
# from crewai import Task
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
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq as LLM
from langchain_core.runnables.base import Runnable
from langchain_core.outputs.llm_result import LLMResult
from litellm import completion

# llm = dict(
#         provider="groq",
#         config=dict(
#             model="groq/llama3-8b-8192",  # Specify the Groq model
#             api_key=os.getenv("GROQ_API_KEY"),  # Use the Groq API key from the environment
#             temperature=0.0,
#         )
# )

# ------ Defining the llm using the LLM class----------

llm= LLM(
    model="groq/llama3-8b-8192",
    temperature=0.0,
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    

)
# ---------- Tried changing the model -> didnt work -----------------------
# llm = LLM(model="gpt-4")
# -------------------------------------------------------------------------

# --- Making the LLM runnable -----------
class RunnableLLM(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input: dict) -> dict:
        # Extract the prompt or query from the input
        prompt = input.get("query")  # Assume 'query' is the input key
        # Generate a response using the LLM's generate method
        response = self.llm.generate(prompt)
        return {"response": response}
    
    # def invoke(self, input, config=None, **kwargs) -> LLMResult:
    #     """
    #     Handles invocation of the LLM with input and additional configurations.
    #     """
    #     # Extract the `stop` parameter if it exists
    #     stop = kwargs.pop('stop', None)

    #     # Generate the response, handling the `stop` argument if provided
    #     if stop:
    #         response = self.llm.generate(input, stop=stop, **kwargs)
    #     else:
    #         response = self.llm.generate(input, **kwargs)

    #     # Convert response to LLMResult format
    #     return LLMResult(generations=[[{"text": response}]])

# Wrap your LLM
runnable_llm = RunnableLLM(llm)

# cohere_api_key = os.getenv("COHERE_API_KEY")

# Input/Output Models

# class ParseAndIndexFilesInput(BaseModel):
#     folder_path: str

# class ParseAndIndexFilesOutput(BaseModel):
#     vectorstore: Any

# class QueryIndexInput(BaseModel):
#     vectorstore: Any  # Replace `Any` with a specific type if available
#     query: str

# class QueryIndexOutput(BaseModel):
#     response: str



# class ParseAndIndexFiles():

#     name = "parse_and_index_files"
#     description = "Parse files in a folder and create an indexed vector store for retrieval."
#     input_model = ParseAndIndexFilesInput
#     output_model = ParseAndIndexFilesOutput



def _extract_text_from_file( file_path):

        _, ext = os.path.splitext(file_path)
        text = ""
        try:
            # PDF
            if ext == ".pdf":
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"

            #DOCX
            elif ext == ".docx":
                if not os.path.basename(file_path).startswith("~$"):
                    try:
                        doc = Document(file_path)
                        for para in doc.paragraphs:
                            text += para.text + "\n"
                    except Exception as e:
                        print(f"Error extracting text: {e}")
                else:
                    print("Skipping temporary file.")

            # TXT
            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as txt_file:
                    text = txt_file.read()
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
        return text





