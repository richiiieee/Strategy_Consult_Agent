from crewai import Crew,Process
from agents import document_query_agent,parse_and_index_agent #,document_fetch_agent,monitor_directory_agent
from tasks import document_query_task,parse_and_index_task #,document_fetch_task,monitor_directory_task
from flow_functions import groq_llm #,runnable_llm
# from langchain.chains import RetrievalQA


# --- Crew ---
crew = Crew(
    agents=[parse_and_index_agent,document_query_agent] ,
    tasks=[parse_and_index_task,document_query_task] ,
    process=Process.sequential,
    max_rpm = 10
)


if __name__ == "__main__":
    while True:
        folder_path = r"C:\D Drive\Richa\NuMindsAI\Projects\CREWAIPROJECT\strategy_consult_agent\new_dir\test_folder"
        # vector_store = parse_and_index.run(path)
        # print(type(folder_path))
        customer_question = "Give a summary of the contents in the directory"  #input("User : \n")
        if customer_question.lower() == 'quit':
            break
        # print("VC : ",vector_store)
        inputs={  
            "folder_path": folder_path,
            "query": customer_question,
        }
        
        print("Inputs passed to kickoff: ", inputs)
        result = crew.kickoff(inputs)
        