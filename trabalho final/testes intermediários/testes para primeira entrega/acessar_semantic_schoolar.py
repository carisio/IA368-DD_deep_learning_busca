import requests
import json
import pickle

# https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_get_paper

# Restrições:
#   limit tem que ser menor que 100
#   fieldsOfStudy (ex.: Engineering )
def search_by_keywords(query,
                       fields='url,title,venue,year,authors,abstract,openAccessPdf,citationCount,referenceCount,publicationTypes,journal,tldr,publicationDate',
                       fieldsOfStudy='Computer Science',
                       year='2020-2023',
                       openAccessPdf=True,
                       offset=0,
                       limit=100):
    query_openaccess = '&openAccessPdf' if openAccessPdf else ''
    url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields={fields}&fieldsOfStudy={fieldsOfStudy}&year={year}{query_openaccess}&offset={offset}&limit={limit}'
    return requests.get(url).json()


def salvar_todos_artigos(query="neural+information+retrieval", nome_arquivo='artigos.pkl', year='2020-2023', fieldsOfStudy="Computer Science"):
    offset = 0
    limit = 100
    total_artigos_retornado = 1
    todos_artigos = []
    
    while offset < total_artigos_retornado and (offset + limit < 10000):
        print(f'Pesquisando {offset} até {offset+limit} de {total_artigos_retornado}')
        retorno = search_by_keywords(query, fieldsOfStudy=fieldsOfStudy, year=year, offset=offset, limit=limit)
        
        todos_artigos.extend(retorno['data'])
        total_artigos_retornado = retorno['total']
        offset += limit
     
    with open(nome_arquivo, 'wb') as f:
        pickle.dump(todos_artigos, f)
          
    return todos_artigos


query = "neural information retrieval models sparse dense architectures"
#artigos_2019_2023 = salvar_todos_artigos(query, "2019-2023 neural information retrieval models sparse dense architectures.pickle", year='2019-2023')
#artigos_2020_2023 = salvar_todos_artigos(query, "2020-2023 neural information retrieval models sparse dense architectures.pickle", year='2020-2023')

#for i, artigo in enumerate(artigos):
#    print(f"{i}: {artigo['title']}")

with open("2020-2023 neural information retrieval models sparse dense architectures.pickle", "rb") as f:
    artigos_2020_2023 = pickle.load(f)

#with open("2019-2023 neural information retrieval models sparse dense architectures.pickle", "rb") as f:
#    artigos_2019_2023 = pickle.load(f)
    
#len(artigos_2020_2023)
#len(artigos_2019_2023)


# Testes com top2vec
from top2vec import Top2Vec
# https://github.com/ddangelov/Top2Vec

from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

model = Top2Vec(documents=newsgroups.data[:50], embedding_model='distiluse-base-multilingual-cased')

artigos_2020_2023[0]['tldr']['text']
model = Top2Vec(documents=[artigos_2020_2023[0]['tldr']['text']], embedding_model='distiluse-base-multilingual-cased')


fields='url,title,venue,year,authors,abstract,openAccessPdf,citationCount,referenceCount,publicationTypes,journal,tldr,publicationDate'
query = 'Quality biased ranking of web documents'
url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields={fields}&year=2011&offset=0&limit=5'
resultado = requests.get(url).json()
for artigo in resultado['data']:
    print(artigo['title'])