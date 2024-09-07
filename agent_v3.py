import json
from crewai import Agent, Task, Crew
from crewai.process import Process
from crewai_tools import BaseTool, SerperDevTool
from langchain_groq import ChatGroq
import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# llama-3.1-70b-versatile
# llama3-70b-8192
llm_groq = ChatGroq(temperature=0,groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")



agent_collector = Agent(
    role="Tourism Research Specialist",
    goal=(
        "Trouve et liste de manière exhaustive tous les sites touristiques du Bénin en utilisant plusieurs sources, y compris les sites web officiels, blogs de voyages, forums de discussions, réseaux sociaux, "
        "et autres plateformes pertinentes. Inclut des informations détaillées sur les parcs nationaux, "
        "monuments historiques, musées, plages, sites culturels et religieux et tout autre lieu d'intérêt."
    ),
    backstory=(
        "Tu es un expert en recherche des sites touristiques du Bénin avec une connaissance approfondie des meilleures sources d'information."
        "Ta tâche est d'identifier et de lister tous les sites touristiques du Bénin, en t'assurant d'utiliser des sources variées pour une couverture complète."
    ),
    llm=llm_groq,
    verbose=True,
    allow_delegation=False
)


manager = Agent(
    role="Project Manager",
    goal=(
        "valider les données collectées par l'agent Tourism Research Specialist relatif a propos des sites touristiques du Bénin"
    ),
    backstory=(
        "Tu es un expert en validation de données touristique"
        "Ta tâche est de valider les données collecté par le Tourism Research Specialist pour t'assurer qu'il s'agit réellement de site touristique du Bénin."
    ),
    llm=llm_groq,
    verbose=True,
    allow_delagation=True
)

task_collector = Task(
    agent=agent_collector,
    description=(
        "Effectue une recherche approfondie pour identifier tous les sites touristiques du Bénin en t'appuyant sur plusieurs sources (sites gouvernementaux, blogs, forums, réseaux sociaux, etc.). "
        "Inclure pour chaque site : son nom, son type (parc, monument, musée, plage, etc.), son adresse complète, "
        "les coordonnées GPS sous forme de lien cliquable, une brève description historique ou culturelle, "
        "les activités disponibles sur le site, et toute information sur des événements ou festivals qui y sont organisés."
    ),
    expected_output=(
        "Un objet JSON avec une clé 'sites touristiques', contenant une liste de sites touristiques. "
        "Chaque site doit avoir les champs : nom, type, adresse, coordonnées GPS, description, "
        "activités disponibles, événements associés, et la source de l'information (site gouvernemental, blog, etc.)."
    )  
)


crew = Crew(
    tasks=[task_collector],
    agents=[agent_collector],
    manager_agent=manager,
    verbose=True,
    process=Process.hierarchical,
    full_output=True,
    manager_llm=llm_groq
)

result = crew.kickoff(inputs={"search_query":"Tous les sites touristiques du Bénin"})
print(result)