import warnings

warnings.filterwarnings("ignore")
from fileinput import filename

warnings.filterwarnings("ignore")
import json
from crewai import Agent, Task, Crew, process
from crewai_tools import BaseTool, SerperDevTool
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import os
import pandas as pd


os.environ["OPENAI_API_KEY"] = "NA"

# llm = ChatOllama(model="llama3.1",base_url="http://localhost:11434")
llm_groq = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.1-70b-versatile",
)


class SaveCSVTool(BaseTool):
    name: str = "SaveCSVTool"
    description: str = "Tool to save provided data into a CSV file."

    def _run(self, data: dict, filename: str = "data_sites_touristiques.csv") -> str:
        df = pd.DataFrame(data["places_interessants"])
        df.to_csv(filename, index=False)
        return f"CSV saved as {filename}"


agent_places = Agent(
    role="Search Tourist Places Specialist",
    goal="Trouve et liste tous les lieux publics, sites touristiques et monuments intéressants pour les touristes au Bénin",
    backstory="Tu es un expert en recherche de lieux d'intérêt pour les touristes au Bénin. Ta tâche est d'identifier et de lister tous les lieux publics, sites touristiques et monuments intéressants, en incluant des descriptions, des images ou des vidéos, et les coordonnées GPS.",
    verbose=True,
    llm=llm_groq,
    tools=[SaveCSVTool(result_as_answer=True)],
)


task_places = Task(
    description=(
        "Rechercher et lister tous les lieux publics, sites touristiques, monuments et autres endroits intéressants pour les touristes au Bénin. "
        "Inclure la description, les images ou vidéos sous forme de liens cliquables, l'adresse du lieu et les coordonnées GPS sous forme de lien cliquable."
    ),
    expected_output="un objet JSON avec pour clé 'places_interessants' contenant les détails demandés",
    agent=agent_places,
)

crew = Crew(
    agents=[agent_places],
    tasks=[task_places],
    verbose=True,
)

result = crew.kickoff(
    inputs={
        "search_query": "Tous lieux publics, sites touristiques, monuments et autres endroits intéressants du Bénin"
    }
)

print(result)
