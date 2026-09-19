import csv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

chunk_sizes = [256]
chunk_overlaps=[0,25,50,100]
similarity_top_ks = [1,3,5,10]
perguntas=[
    "Qual é o objetivo e aplicação estabelecidos pela NR-26?",
    "Quais as diretrizes para sinalização por cor e a limitação do seu uso nos ambientes de trabalho?",
    "Qual sistema internacional deve ser utilizado como critério para classificação dos perigos de produtos químicos?",
    "Na ausência de uma lista nacional de classificação harmonizada de substâncias perigosas, qual fonte pode ser utilizada?",
    "Quais são os seis elementos obrigatórios que compõem a rotulagem preventiva do produto químico classificado como perigoso?",
    "Conforme o GHS, como deve ser a estruturação da rotulagem preventiva de produtos químicos que NÃO são classificados como perigosos?",
    "Considerando os subitens 26.4.2.1, 26.4.2.1.1 e 26.4.2.2 - quais produtos estão dispensados do cumprimento das exigências de rotulagem preventiva?",
    "Quem é responsável por elaborar e disponibilizar a ficha com dados de segurança do produto químico perigoso?",
    "Para misturas, quais substâncias devem ter seu nome e concentração explicitados na ficha com dados de segurança?",
    "Quais são os temas obrigatórios sobre os quais os trabalhadores devem receber treinamento em relação aos produtos químicos?"
]


Settings.llm = Ollama(model="llama3.1", request_timeout=60.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text",temperature=0.1)

print("Iniciando a base de conhecimento.")
documents = SimpleDirectoryReader("dados").load_data()

with open("resultados_comparativos.csv", "w", newline="", encoding="utf-8-sig") as arquivo:
    # colunas = ["chunk_size", "chunk_overlap", "similarity_top_k", "pergunta", "resposta"]
    # tabela = csv.DictWriter(arquivo, fieldnames=colunas)
    # tabela.writeheader()

    for chunk_size in chunk_sizes:
        for chunk_overlap_value in chunk_overlaps:
            print(
                f"\n\n============================================================"
                f"\nIndexando: chunk_size={chunk_size}, "
                f"chunk_overlap={chunk_overlap_value}"
            )
            Settings.node_parser = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap_value,
            )
            index = VectorStoreIndex.from_documents(documents)

            for similarity in similarity_top_ks:
                query_engine = index.as_query_engine(similarity_top_k=similarity)

                for pergunta in perguntas:
                    
                    resposta = query_engine.query(pergunta)
                    print(
                            f"\n\nRespondendo: chunk_size={chunk_size}, "
                            f"chunk_overlap={chunk_overlap_value}, "
                            f"similarity_top_k={similarity}"
                            f"\npergunta={pergunta}"
                            f"\nresposta={resposta}"
                        )
                    # tabela.writerow(
                    #     {
                    #         "chunk_size": chunk_size,
                    #         "chunk_overlap": chunk_overlap_value,
                    #         "similarity_top_k": similarity,
                    #         "pergunta": pergunta,
                    #         "resposta": str(resposta),
                    #     }
                    # )

print("Fim do processo.")

