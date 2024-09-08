import warnings
import json
import os
import pandas as pd
from langchain_ollama import ChatOllama
from crewai import Agent, Task, Crew, process 
from crewai_tools import BaseTool, SerperDevTool
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")

os.environ["OPENAI_API_KEY"] = "NA"

# Configuration du modèle
llm = ChatOllama(model="llama3.1",base_url="http://localhost:11434")
llm_groq = ChatGroq(temperature=0, groq_api_key="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", model_name="llama-3.1-70b-versatile")

# Outil pour enregistrer les données en JSON
class SaveJSONTool(BaseTool):
    name: str = "SaveJSONTool"
    description: str = "Tool to save provided data into a JSON file."

    def _run(self, data: dict, filename: str = "sites_touristiques_description.json") -> str:
        with open(filename, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return f"JSON saved as {filename}"

# Premier agent : Recherche et liste les sites touristiques du Bénin
agent_liste = Agent(
    role="Rechercheur de Sites Touristiques",
    goal="Trouver et lister tous les sites touristiques du Bénin",
    backstory="Tu es un expert en recherche de sites touristiques au Bénin. Ta tâche est de lister tous les sites touristiques du Bénin avec leur adresse et leurs coordonnées GPS.",
    verbose=True,
    llm=llm_groq,
    tools=[SaveJSONTool(result_as_answer=True)]
)

task_liste = Task(
    description="Rechercher et lister tous les sites touristiques du Bénin avec leurs adresses et coordonnées GPS.",
    expected_output="Un objet JSON avec une clé 'sites_touristiques'.",
    agent=agent_liste
)

# Deuxième agent : Crée des descriptions détaillées pour chaque site
agent_description = Agent(
    role="Créateur de Descriptions",
    goal="Créer des descriptions détaillées pour chaque site touristique du Bénin.",
    backstory="Tu es un expert en rédaction. Ta tâche est de générer des descriptions de 200 mots pour chaque site touristique listé.",
    verbose=True,
    llm=llm_groq,
    tools=[SaveJSONTool(result_as_answer=True)]
)

task_description = Task(
    description=(
        "Utiliser les noms et descriptions des sites touristiques du Bénin pour créer un article de 200 mots pour chaque site."
    ),
    expected_output="Un fichier JSON avec les descriptions des sites.",
    agent=agent_description
)

# Crew pour gérer les tâches et agents
crew = Crew(
    agents=[agent_liste, agent_description],
    tasks=[task_liste, task_description],
    verbose=True,
)

# Exécution de la première tâche (listing des sites)
result_liste = crew.kickoff(inputs={"search_query": "Tous les sites touristiques du Bénin"})
print(result_liste)

# Exécution de la deuxième tâche (création des descriptions)
if "sites_touristiques" in result_liste:
    site_data = result_liste["sites_touristiques"]
    result_description = crew.kickoff(inputs={"sites_touristiques": site_data})
    print(result_description)
else:
    print("Erreur lors de la récupération des sites touristiques.")
