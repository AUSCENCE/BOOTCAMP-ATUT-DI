

import warnings
from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")

os.environ["OPENAI_API_KEY"] = "NA"

llm_groq = ChatGroq(temperature=0, groq_api_key="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", model_name="llama-3.1-70b-versatile")

class ScrapeLocationTool(BaseTool):
    name: str = "ScrapeLocationTool"
    #description: str = "Tool to scrape and find location data (coordinates, address) for tourism sites."

    def _run(self, site_name: str) -> dict:
        search_url = f"https://www.google.com/search?q={site_name}+tourism+site+location"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        location_data = {}
        location_data["name"] = site_name
        location_data["address"] = soup.find('span', {'class': 'LrzXr'}).text if soup.find('span', {'class': 'LrzXr'}) else "No address found"
        location_data["coordinates"] = soup.find('div', {'data-attrid': 'kc:/location/location:coordinates'}).text if soup.find('div', {'data-attrid': 'kc:/location/location:coordinates'}) else "No coordinates found"

        return location_data

class UpdateCSVTool(BaseTool):
    name: str = "UpdateCSVTool"
   # description: str = "Tool to read, update and save location data into a CSV file."

    def _run(self, filename: str, updated_data: dict) -> str:
        df = pd.read_csv(filename)

        df['Address'] = df['Name'].map(lambda name: updated_data[name]['address'] if name in updated_data else "Not found")
        df['Coordinates'] = df['Name'].map(lambda name: updated_data[name]['coordinates'] if name in updated_data else "Not found")

        df.to_csv(filename, index=False)
        return f"CSV updated and saved as {filename}"

agent = Agent(
    role="Data Enrichment Specialist",
    goal="Lire le fichier CSV contenant les noms des sites touristiques, rechercher leur localisation et mettre à jour le fichier.",
    backstory="Tu es un expert en enrichissement de données touristiques. Ta tâche est de compléter les informations de localisation pour chaque site touristique.",
    verbose=True,
    llm=llm_groq,
    tools=[ScrapeLocationTool(result_as_answer=True), UpdateCSVTool(result_as_answer=True)]
)

task = Task(
    description=(
        "Lisez un fichier CSV contenant les noms et descriptions des sites touristiques, "
        "recherchez des informations sur la localisation comme l'adresse et les coordonnées GPS, "
        "puis mettez à jour le fichier CSV."
    ),
    expected_output="Mise à jour du fichier CSV avec des informations sur l'adresse et les coordonnées GPS.",
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
)

filename = "sites_touristiques.csv"

initial_data = pd.read_csv(filename)
names_to_search = initial_data["Name"].tolist()

result = crew.kickoff(inputs={"search_query": names_to_search, "csv_file": filename})

print(result)