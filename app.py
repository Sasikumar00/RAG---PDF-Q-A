from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

#Keys
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = "45fbc800-5f9a-4fca-89d1-fe0915503649"

#Initialize
app = Flask(__name__)
# llm = Ollama(model="llama2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    is_separator_regex=False
)

# - Ollama embedding
embeddings = OllamaEmbeddings(
    model="llama2",
)

# - Pinecone
index_name="ragg-app"
# vectorStore = PineconeVectorStore(index_name=index_name,embedding=embeddings)


@app.route('/ai', methods=['POST'])
def getResponse():
    #Get the document from user
    req_body = request.json
    query = req_body.get("query")

    #Load the PDF using PDFLoader
    loader = PyPDFDirectoryLoader("docs/")
    loadedPdf = loader.load()

    #Split the PDF into document using RecursiveCharacterTextSplitter
    documents = text_splitter.split_documents(loadedPdf)

    #Create embeddings for each document
    #Insert the embeddings into the pinecone DB with metadata
    PineconeVectorStore.from_documents(documents,embedding=embeddings,index_name=index_name)

    #Take input question from user
    
    #Based on the input question, get the related data for the question from the pinecone DB

    #Attach the context to the promt template by langchain
    # response = llm.invoke(query)

    #Get a response from the model 

    #Return the result to the frontend
    return 'success'

def start_app():
    app.run(port=8080,debug=True)

if __name__ == "__main__":
    start_app()