from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# 1. Configurar os modelos locais do Ollama no LlamaIndex
# Substitua pelo modelo exato que você tem baixado (ex: llama3.1, mistral)
Settings.llm = Ollama(model="llama3.1", request_timeout=60.0)

# Configurar o modelo de embeddings para vetorizar o texto da norma
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

print("Carregando o documento da norma técnica...")
# 2. Carregar o documento de texto que está dentro da pasta 'dados'
documents = SimpleDirectoryReader("dados").load_data()

print("Criando o banco de dados vetorial (Indexando)...")
# 3. Fragmentar o texto e criar o índice vetorial na memória
index = VectorStoreIndex.from_documents(documents)

# 4. Criar o motor de consulta (Query Engine)
query_engine = index.as_query_engine()

print("\n--- RAG Pronto para uso! ---")

# 5. Loop para fazer perguntas no terminal
while True:
    pergunta = input("\nDigite sua pergunta sobre a norma (ou 'sair'): ")
    if pergunta.lower() == 'sair':
        break
        
    print("Buscando resposta...")
    resposta = query_engine.query(pergunta)
    print("\nResposta da IA:")
    print(resposta)