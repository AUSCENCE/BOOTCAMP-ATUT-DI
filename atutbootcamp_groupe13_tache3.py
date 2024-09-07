import warnings
import pandas as pd
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
from langchain_groq import ChatGroq
from langchain.tools import DuckDuckGoSearchRun

warnings.filterwarnings("ignore")

# Ensure the environment uses UTF-8
import locale
locale.setlocale(locale.LC_ALL, 'C.UTF-8')

# Initialize the language model
llm_groq = ChatGroq(temperature=0, groq_api_key="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", model_name="llama-3.1-70b-versatile")

class ReadCSVTool(BaseTool):
    name: str = "ReadCSVTool"
    description: str = "Tool to read data from the initial CSV file."

    def _run(self, filename: str) -> str:
        try:
            df = pd.read_csv(filename, encoding="utf-8")
            return df.to_json(orient="records")
        except Exception as e:
            return f"Error reading CSV: {str(e)}"

class InternetSearchTool(BaseTool):
    name: str = "InternetSearchTool"
    description: str = "Tool to perform internet searches."

    def _run(self, query: str) -> str:
        search = DuckDuckGoSearchRun()
        return search.run(query)

class LocationSearchTool(BaseTool):
    name: str = "LocationSearchTool"
    description: str = "Tool to search for location information using internet search."

    def _run(self, site_name: str) -> str:
        search = DuckDuckGoSearchRun()
        query = f"{site_name} coordinates Google Maps"
        results = search.run(query)
        # Note: This is a simplified version. In a real scenario, you'd need to parse the results
        # to extract coordinates, which might require more sophisticated processing.
        return f"Search results for {site_name}: {results}"

class UpdateCSVTool(BaseTool):
    name: str = "UpdateCSVTool"
    description: str = "Tool to update the CSV file with new information."

    def _run(self, filename: str, updated_data: str) -> str:
        try:
            df = pd.read_csv(filename, encoding="utf-8")
            
            updated_data_list = eval(updated_data)
            for item in updated_data_list:
                name = item['nom']
                df.loc[df['nom'] == name, 'Adresse'] = item.get('Adresse', '')
                df.loc[df['nom'] == name, 'Latitude'] = item.get('Latitude', '')
                df.loc[df['nom'] == name, 'Longitude'] = item.get('Longitude', '')

            df.to_csv(filename, index=False, encoding="utf-8")
            return f"CSV updated and saved as {filename}"
        except Exception as e:
            return f"Error updating CSV: {str(e)}"

agent = Agent(
    role="Data Enrichment Specialist",
    goal="Read the CSV file containing tourist site information, perform additional research, and update the file with new information.",
    backstory="You are an expert in enriching tourism data. Your task is to complete location information for each tourist site using existing data and conducting additional research.",
    verbose=True,
    llm=llm_groq,
    tools=[ReadCSVTool(), InternetSearchTool(), LocationSearchTool(), UpdateCSVTool()]
)

task = Task(
    description=(
        "1. Read the 'sites_touristiques_benin.csv' file using ReadCSVTool. "
        "2. For each site, analyze the name, description, and article. "
        "3. Use InternetSearchTool to find additional information if necessary. "
        "4. Use LocationSearchTool to obtain the precise address and GPS coordinates. "
        "5. Compile all obtained information. "
        "6. Use UpdateCSVTool to update the CSV file with the new information, including GPS coordinates."
    ),
    expected_output="An updated CSV file with precise addresses and GPS coordinates for each tourist site.",
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential
)

# Launch the agent to perform additional research
result = crew.kickoff()

print(result)