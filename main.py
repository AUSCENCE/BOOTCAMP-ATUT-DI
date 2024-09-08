import warnings
import json
import os
import pandas as pd

warnings.filterwarnings("ignore")

from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from langchain_groq import ChatGroq


os.environ["GROQ_API_KEY"] = "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"


llm_groq = ChatGroq(
    temperature=0,
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama-3.1-70b-versatile",
)


class SaveJSONTool(BaseTool):
    name: str = "SaveJSONTool"
    description: str = "Tool to save provided data into a JSON file."

    def _run(self, data: dict, filename: str = "data_enriched.json") -> str:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return f"JSON saved as {filename}"


agent = Agent(
    role="Tourism Data Enricher",
    goal="Enrichir les données des sites touristiques du Bénin contenues dans les données fournies en entrée.",
    backstory="Tu es un expert en gestion de données géographiques et touristiques du Bénin.",
    verbose=True,
    llm=llm_groq,
    tools=[SaveJSONTool(result_as_answer=True)],
)


task = Task(
    description=(
        "Prendre les données d'entrée et enrichir chaque site touristique avec des informations géographiques "
        "et économiques, telles que la position GPS, la distance de la capitale, et le coût du voyage."
    ),
    expected_output="Un objet JSON en réponse avec les informations enrichies sous la clé 'sites_touristiques'.",
    agent=agent,
)


crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
)


result = crew.kickoff(
    inputs={
        "sites_touristiques": [
            {"nom": "Parc National de la Pendjari", "ville": "Tanguiéta"},
        ]
    }
)


print(result)
