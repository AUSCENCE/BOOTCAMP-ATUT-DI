
import warnings
from fileinput import filename
warnings.filterwarnings("ignore")
import json
from crewai import Agent, Task, Crew,process
from crewai_tools import BaseTool,SerperDevTool
from  langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import os
import pandas as pd



os.environ["OPENAI_API_KEY"] ="NA"

#llm = ChatOllama(model="llama3.1",base_url="http://localhost:11434")
llm_groq = ChatGroq(temperature=0,groq_api_key="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",model_name="llama-3.1-70b-versatile")

class SaveCSVTool(BaseTool):
    name: str ="SaveCSVTool"
    description: str ="Tool to save provided data into a CSV file."

    def _run(self,data:dict, filename: str ="data_future_events.csv") -> str:
        df = pd.DataFrame(data["future_events"])
        df.to_csv(filename,index=False)
        f = open(filename+".json", "w")
        f.write(json.dumps(data))
        f.close()
        return "CSV saved as {filename}"

agent= Agent(
    role="Search Specialist",
    goal="Trouve et liste de tous les évènements futures du Bénin",
    backstory="Tu es un expert en recherche des des évènements futures du Bénin.Ta tâche est d'identifier et de lister tous les évènements futures du Bénin.",
    verbose=True,
    llm=llm_groq,
    tools = [SaveCSVTool(result_as_answer=True)]
)


task= Task(
    description=(
        "rechercher et lister tous les évènements futures du Bénin"
        "Inclure l'adresse de déroulement et les coordonnées GPS sous forme de lien cliquable"
        "Inclure une description complète"
        "Inclure les urls"
    ),
    expected_output="un objet json comme réponse avec pour clé 'future_events",
    agent=agent
)

crew = Crew(
    agents= [agent],
    tasks= [task],
    verbose=True,
)


result = crew.kickoff(inputs={"search_query":"Tous les évènements futures du Bénin"})


print(result)



from minio import Minio
from dotenv import load_dotenv
import os

load_dotenv()

LOCAL_FILE_PATH = os.environ.get('LOCAL_FILE_PATH')
#ACCESS_KEY = os.environ.get('Q3AM3UQ867SPQQA43P2F')
#SECRET_KEY = os.environ.get('zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG')
MINIO_API_HOST = "play.min.io"

utils = Minio("play.min.io", access_key="Q3AM3UQ867SPQQA43P2F", secret_key="zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG", secure=True)

#targetBucket = "Groupe1Tache1"
targetBucket = "test"

found = utils.bucket_exists(targetBucket)
if not found:
  utils.make_bucket(targetBucket)
else:
  print("Bucket already exists")

utils.fput_object(targetBucket, "groupe1_tache1_evenements_futurs_benin.json", "/content/evenements_futurs_benin.csv.json")

'''utils = PythonMinIOUtils(
        endpoint="play.min.io",
        access_key="Q3AM3UQ867SPQQA43P2F",
        secret_key="zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG",
        secure=True,
        bucket_name="test")'''
    #utils.download_file('graph_true_data.pkl', 'graph_true_data.pkl')