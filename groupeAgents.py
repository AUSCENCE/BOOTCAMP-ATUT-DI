import warnings
import json
import os
import pandas as pd
from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")

os.environ["OPENAI_API_KEY"] = "NA"

llm_groq = ChatGroq(
    temperature=0, 
    groq_api_key="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", 
    model_name="llama-3.1-70b-versatile"
)

search_agent = Agent(
    role="Search Specialist",
    goal="Trouver et lister tous les sites touristiques du Bénin",
    backstory="Tu es un expert en recherche des sites touristiques du Bénin. Ta tâche est d'identifier et de lister tous les sites touristiques du Bénin.",
    verbose=True,
    llm=llm_groq,
)

task_search = Task(
    description=(
        "Rechercher et lister tous les sites touristiques du Bénin. "
        "Inclure l'adresse du site, les coordonnées GPS sous forme de lien cliquable, et une courte description."
    ),
    expected_output="Un objet JSON avec les sites touristiques, incluant les noms et descriptions.",
    agent=search_agent
)

content_writer_agent = Agent(
    role="Content Writer",
    goal="Rédiger des articles de 200 mots pour chaque site touristique identifié en utilisant uniquement le nom et la description.",
    backstory="Tu es un expert en rédaction. Ton objectif est de créer un article en utilisant le nom et la description des sites touristiques fournis par l'agent de recherche.",
    verbose=True,
    llm=llm_groq
)

def create_content_task(site_data):
    site_name = site_data.get('nom', 'un site touristique')
    description = site_data.get('description', 'Aucune description disponible')

    return Task(
        description=f"Rédiger un article de 200 mots pour le site touristique '{site_name}'. "
                    f"Utilise uniquement cette description fournie : {description}.",
        expected_output="Un article de 200 mots.",
        agent=content_writer_agent
    )

crew_search = Crew(
    agents=[search_agent],
    tasks=[task_search],
    verbose=True,
)

search_result = crew_search.kickoff(inputs={"search_query": "Tous les sites touristiques du Bénin"})

print("Résultat de la recherche :")
print(search_result)

if hasattr(search_result, 'output'):
    sites_touristiques = search_result.output.get('sites_touristiques', [])
else:
    print("Aucun site touristique trouvé.")
    sites_touristiques = []

if not sites_touristiques:
    print("Aucun site touristique trouvé.")
else:
    updated_sites = []

    for site in sites_touristiques:
        task_content = create_content_task(site)
        crew_content = Crew(
            agents=[content_writer_agent],
            tasks=[task_content],
            verbose=True,
        )

        article_result = crew_content.kickoff(inputs={"site_data": site})
        article = article_result

        site_with_article = site.copy()
        site_with_article["article"] = article

        updated_sites.append(site_with_article)

    final_result = {"sites_touristiques": updated_sites}

    with open("sites_touristiques_benin.json", "w", encoding="utf-8") as json_file:
        json.dump(final_result, json_file, ensure_ascii=False, indent=4)

    print("Les données ont été enregistrées dans sites_touristiques_benin.json")
