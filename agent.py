import warnings
import json
import os
import pandas as pd
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from langchain_ollama import ChatOllama

# Ignorer les avertissements pour une exécution plus propre (à n'utiliser que si vous comprenez les risques)
warnings.filterwarnings("ignore")

# Clé API et configuration Ollama (assurez-vous que l'API fonctionne sur cette URL)
os.environ["OPENAI_API_KEY"] = "NA"  # Modifiez ou supprimez si ce n'est pas nécessaire
llm_ollama = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

# Outil pour sauvegarder les résultats dans un fichier JSON
class SaveJSONTool(BaseTool):
    name: str = "SaveJSONTool"
    description: str = "Outil pour sauvegarder les données dans un fichier JSON."

    def _run(self, data: dict, filename: str = "data_sites_touristiques_enrichi.json") -> str:
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            return f"JSON saved as {filename}"
        except Exception as e:
            return f"Failed to save JSON: {str(e)}"

# Fonction pour charger les données depuis le fichier JSON
def load_json_data(filepath: str) -> dict:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {filepath}.")
        return {}

# Charger les données JSON initiales
json_data = load_json_data("sites_touristiques.json")

# Vérification que les données JSON sont bien chargées
if not json_data:
    print("No data loaded. Exiting...")
    exit(1)

# Création de l'agent IA pour enrichir les données
agent = Agent(
    role="Search Specialist",
    goal="Enrichir les informations des sites et événements touristiques du Bénin",
    backstory="Tu es un expert en recherche des sites touristiques du Bénin. "
              "Ta tâche est de trouver les adresses et coordonnées GPS des sites et événements touristiques.",
    verbose=True,
    llm=llm_ollama,
    tools=[SaveJSONTool()]  # Suppression de 'result_as_answer' si non nécessaire dans 'BaseTool'
)

# Définition de la tâche
task = Task(
    description=(
        "Récupérer les informations sur les sites et événements touristiques du Bénin. "
        "Pour chaque site ou événement, trouver son adresse et ses coordonnées GPS."
    ),
    expected_output="Un objet JSON enrichi avec les adresses et les coordonnées GPS des sites touristiques",
    agent=agent
)

# Initialisation de l'équipe d'agents pour exécuter la tâche
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
)

# Démarrage de l'agent pour enrichir les données
result = crew.kickoff(inputs={"search_query": json_data})

# Affichage du résultat de la tâche (ou vous pourriez le sauvegarder)
print(result)

# Sauvegarde du résultat dans un fichier JSON
save_tool = SaveJSONTool()
save_tool._run(data=result, filename="data_sites_touristiques_enrichi.json")
