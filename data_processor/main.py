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

class DataEnrichment(BaseTool):
    name: str ="DataEnrichment"
    description: str ="Data enrichment tool"

    def _run(self,data:dict, url_csv:str = "data.csv", filename: str ="articles_data_sites_touristiques.csv") -> str:
        df = pd.read_csv(url_csv)
        return "JSON saved as {filename}"

    def enrichment():
        pass


agent= Agent(
    role="Enrichment Specialist",
    goal="Produit un article détaillé sur le nom de chaque site ou évènement touristique du Bénin répertorié",
    backstory="Tu es un expert en création d'articles sur les sites et evènements touristiques du Bénin. Ta tâche est de produire un article détaillé sur le nom de chaque site ou évènement touristique du Bénin répertorié",
    verbose=True,
    llm=llm_groq,
    tools = [DataEnrichment(result_as_answer=True)]
)


task= Task(
    description=(
        "Analyse le nom et la description des sites touristiques fournis dans le fichier CSV ({file_data_csv})"
        "Pour produire un article détaillé sur chacun d'eux"
    ),
    expected_output="un objet json comme réponse avec pour clé nom ayant comme valeur le nom d'un site fournis dans le fichier CSV ({file_data_csv}) et article ayant pour "
    "valeur le contenu de l'article",
    agent=agent
)

crew = Crew(
    agents= [agent],
    tasks= [task],
    verbose=True,
)

job_crew_works = {
    'file_data_csv':'data.csv'
}

result = crew.kickoff(inputs=job_crew_works)


print(result)