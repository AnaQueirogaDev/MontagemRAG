from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="llama3.1", request_timeout=60.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text",temperature=0.1)
Settings.node_parser = SentenceSplitter(chunk_size=128,chunk_overlap=25)

print("Iniciando a base de conhecimento.")
documents = SimpleDirectoryReader("dados").load_data()

print("Indexando a base de conhecimento.")
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=1)

print("\nBem-vindo ao sistema de perguntas e respostas sobre a NR 26.")

while True:
    pergunta = input("\nDigite sua pergunta (ou 'S' para sair): ")
    if pergunta.upper() == 'S':
        break
        
    print("Conectando com a IA para obter a resposta...")
    resposta = query_engine.query(pergunta)
    print("\nResposta da IA:")
    print(resposta)

