import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv

load_dotenv()

loader = CSVLoader(file_path="INC.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
Você é um assistente virtual de uma construtora de médio porte em Juiz de Fora, chamada JF Emprrendimentos.
Sua função será tirar dúvidas sobre os processos dos setores da empresa.
Vou lhe passar alguns modelos de processos do setor de TI da empresa para que você use como modelo.

Siga as regras abaixo:

1/ Preste atenção somente ao conteúdo útil da mensagem.

2/ Suas respostas devem ser bem similares ou até identicas às enviadas no passado, tanto em termos de comprimento, tom de voz, argumentos lógicos e demais detalhes.

3/Você deve sempre ressaltar como funciona o processo da empresa que se encaixa na dúvida do colaborador.

Aqui está uma dúvida de um novo colaborador.
{message}

Aqui está uma lista de dúvidas realizadas pelos nossos colaboradores anteriormente e a resposta de como ele deve atuar de acordo com os processos da empresa. 
Este histórico de conversa servirá de base para que você compreenda nossos processos e como responder as dúvidas dos colaboradores.
{best_pratice} 

Descreva detalhadamente como eu deveria atuar usando como base as melhores práticas utilizadas na empresa.
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

def main():
    st.set_page_config(
        page_title="Mariantonia", page_icon=":bird:")
    st.header("Mariantonia")
    message = st.text_area("Digite sua Pergunta")

    if message:
        st.write("Buscando Respostas no Sistema")

        result = generate_response(message)

        st.info(result)

if __name__ == '__main__':
    main()