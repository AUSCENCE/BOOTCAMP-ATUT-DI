import warnings
import json
import os
import pandas as pd
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from langchain_ollama import ChatOllama
warnings.filterwarnings("ignore")

# Clé API et configuration Ollama (modifiez en fonction de votre installation locale)
os.environ["OPENAI_API_KEY"] = "NA"
llm_ollama = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

# Outil pour sauvegarder les résultats dans un fichier JSON
class SaveJSONTool(BaseTool):
    name: str = "SaveJSONTool"
    description: str = "Outil pour sauvegarder les données dans un fichier JSON."

    def _run(self, data: dict, filename: str = "data_sites_touristiques_enrichi.json") -> str:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return f"JSON saved as {filename}"

# Charger les données depuis le fichier JSON initial
def load_json_data(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Exemple de structure des données JSON à enrichir
json_data = load_json_data("sites_touristiques.json")

# Agent IA pour effectuer le scraping général
agent = Agent(
    role="Search Specialist",
    goal="Enrichir les informations des sites et événements touristiques du Bénin",
    backstory="Tu es un expert en recherche des sites touristiques du Bénin. "
              "Ta tâche est de trouver les adresses et coordonnées GPS des sites et événements touristiques.",
    verbose=True,
    llm=llm_ollama,
    tools=[SaveJSONTool(result_as_answer=True)]
)

# Définition de la tâche pour l'agent
task = Task(
    description=(
        "Récupérer les informations sur les sites et événements touristiques du Bénin. "
        "Pour chaque site ou événement, trouver son adresse et ses coordonnées GPS."
    ),
    expected_output="Un objet JSON enrichi avec les adresses et les coordonnées GPS des sites touristiques",
    agent=agent
)

# Lancement du scraping par l'agent IA
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
)

# Requête pour commencer la tâche avec l'agent IA
result = crew.kickoff(inputs={"search_query": json_data})

# Sauvegarder les résultats dans un fichier JSON
print(result)