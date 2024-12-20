from crewai import Task
# from tools import ParseAndIndexFilesTool, QueryIndexTool
from agents import document_query_agent,parse_and_index_agent  #,document_fetch_agent,monitor_directory_agent
from tools import QueryIndex,parse_and_index_tool#, monitor_directory_tool


parse_and_index_task = Task(
    description = """You will process and organize data from a {folder_path} containing various files. 
    The task involves extracting text content from these files, splitting the content into smaller, 
    manageable chunks, generating vector embeddings for each chunk, and storing these embeddings in 
    a FAISS vector store. This setup allows you to efficiently search and retrieve information from 
    the processed documents.
    
    Input : {folder_path}
    """,
    expected_output = """ FAISS vector store that includes:

    Indexed embeddings of document chunks.
    Metadata that links each embedding to its original source file.""",
    tools=[parse_and_index_tool],
    agent=parse_and_index_agent,
)

document_query_task = Task(
    description=(
        """
        Your task is to process user queries and retrieve the most relevant information 
        from a collection of documents within a designated folder. You analyze the user’s 
        input : {query}, to understand t heir intent and determine the specific information they are 
        seeking. Using advanced search algorithms, you navigate through the files in the 
        folder to locate relevant content, whether it appears as text, tables, or embedded 
        metadata. Your ability to interpret the context of queries allows you to refine 
        your search and provide accurate results, even when the questions are complex or 
        nuanced. Once the relevant information is identified, you deliver it clearly and 
        concisely, either by highlighting specific sections of documents or by summarizing 
        the content to meet the user’s needs. You adapt to various file formats and handle 
        ambiguous queries with precision, ensuring that your responses are both timely and reliable.
        """
    ),
    expected_output="""
        Provide clear and accurate answers to the customer's questions based on 
        the content of the documents within the directory.
        """,
    tools=[QueryIndex],
    agent=document_query_agent,
)

# --------------------- Monitoring --------------------------------------------
# monitor_directory_task = Task(
#     description = """ Your task is to monitor a specific directory (and its subdirectories, if requested) 
#     for any changes in real time. The monitoring process will continue until explicitly stopped. The tool 
#     will provide a status message once the monitoring has started and also notify the user if an error occurs.
#     You will receive an input dictionary containing:

#     path:       The directory path to monitor. Defaults to the current directory (.) if not provided.
#     recursive:  A boolean value indicating whether to monitor subdirectories as well. Defaults to 
#                 True for recursive monitoring.

#     The tool successfully starts monitoring the directory and subdirectories.
#     If no errors occur, the tool continues running and monitoring changes until manually stopped.
#     """,
#     expected_output = """Return the file containing the names of the files that were updated to the directory.""",
#     tools = [monitor_directory_tool],
#     agent = monitor_directory_agent
# )

# document_fetch_task = Task(
#     description = """ You are tasked with retrieving files from a given directory that 
#     match a specific pattern. Use the fetch_files tool to perform this task. Specify 
#     the file type or pattern you are looking for (e.g., *.txt for text files, **/*.pdf 
#     for PDFs in all subdirectories) and the base directory to search in. The tool will 
#     return a list of all matching files.
    
#     Input:
#     "pattern":str
#     "base_dir":str
     
#     Output:
#     "files": ["list_of_files"] """,
#     expected_output = """ List of all the matching files.""",
#     tools = [fetch_files],
#     agent = document_fetch_agent,
#     context = [monitor_directory_task]
# )
