from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

#Load Documents
pdf_paths = input("Enter PDF paths (comma-separated): ").split(",")

all_docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path.strip())
    all_docs.extend(loader.load())


#Split text
txt_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs=txt_splitter.split_documents(all_docs)

#Embeddings
embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
vector_store=Chroma.from_documents(embedding=embeddings,documents=docs,persist_directory='Sample')

#Retriever
retriever=vector_store.as_retriever()

#Model
llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash')
chain=ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,return_source_documents=True)#Returns sources

chat_history=[]


while True:
  user_input=input("You: ")
  if user_input.lower()=='done':
    break
  response=chain.invoke({
     'question':user_input,
     'chat_history':chat_history
  })
  print('\nBot: ',response['answer'])
  print('\nSources:')
  for doc in response["source_documents"]:
    print(f"- {doc.metadata['source']} | Page {doc.metadata['page']+1}")
  chat_history.append((user_input,response['answer']))
  






