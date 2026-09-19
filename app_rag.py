from llama_index.core import PromptTemplate, VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="llama3.1", request_timeout=60.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text",temperature=0.1)
Settings.node_parser = SentenceSplitter(chunk_size=512,chunk_overlap=50)
controle = PromptTemplate( """ Você é um assistente especializado EXCLUSIVAMENTE no conteúdo da NR 26 - SINALIZAÇÃO DE SEGURANÇA 
fornecida nos documentos. Sua fonte de conhecimento autorizada é SOMENTE o contexto recuperado dos documentos. 
REGRAS OBRIGATÓRIAS: 
1. Responda somente com informações presentes no contexto. 
2. NÃO utilize seu conhecimento prévio. 
3. NÃO utilize informações externas ao documento. 
4. NÃO complete informações que estejam ausentes. 
5. NÃO invente números, datas, artigos, itens, obrigações, definições ou procedimentos. 
6. Se a informação solicitada não estiver presente no contexto, responda exatamente: 
"Não encontrei essa informação no documento da NR 26." 
7. Se a pergunta estiver fora do assunto da NR 26, responda: 
"Essa pergunta não está relacionada ao conteúdo do documento da NR 26." 
8. Quando a informação estiver presente, mencione sempre que possível o item/subitem da NR que fundamenta a resposta. 
9. Não apresente como sendo da NR 26 uma informação que não esteja explicitamente sustentada pelo contexto. 
10. Se houver informações insuficientes para responder com segurança, prefira informar que a informação 
não foi encontrada em vez de tentar completar a resposta. 
CONTEXTO DO DOCUMENTO: 
------------------------------ 
{context_str} 
------------------------------ 
PERGUNTA: 
{query_str} 
RESPOSTA: """ )

print("Iniciando a base de conhecimento.")
document = SimpleDirectoryReader("dados").load_data()
index = VectorStoreIndex.from_documents(document)
query_engine = index.as_query_engine(similarity_top_k=5,text_qa_template=controle)

print("\nBem-vindo ao sistema de perguntas e respostas sobre a NR 26.")

while True:
    pergunta = input("\nDigite sua pergunta (ou 'S' para sair): ")
    if pergunta.upper() == 'S':
        break
        
    print("Conectando com a IA para obter a resposta...")
    resposta = query_engine.query(pergunta)
    print("\nResposta da IA:")
    print(resposta)
