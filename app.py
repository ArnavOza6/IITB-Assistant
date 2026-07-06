import streamlit as st

from dotenv import load_dotenv
load_dotenv()

import os
from groq import Groq
def get_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None

api_key = get_api_key()
groq_client = Groq(api_key=api_key) 

import PyPDF2
import chromadb
from chromadb.utils import embedding_functions

def read_pdf_file(file_path: str):
  text = ""
  with open (file_path,'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text

#CHUNKING FUNCTION

def split_text(text:str, chunk_size: int = 500, chunk_overlap = 50) -> list:
  sentences = text.replace('\n',' ').split('. ')
  chunks = []
  current_chunk = []
  current_size = 0 

  for sentence in sentences: 
    sentence = sentence.strip()
    if not sentence:
      continue

    if not sentence.endswith('.'):
      sentence += '.'

    sentence_size = len(sentence)

    if current_size + sentence_size + (1 if current_chunk else 0) > chunk_size:
      overlap_chunk = []
      overlap_size = 0
      for s in reversed(current_chunk):
         if overlap_size + len(s) + (1 if overlap_chunk else 0) <= chunk_overlap:
            overlap_chunk.insert(0, s)
            overlap_size += len(s) + (1 if overlap_chunk else 0)
         else:
            break
      chunks.append(" ".join(current_chunk))
      current_chunk = overlap_chunk
      current_size = overlap_size
    
    current_size += sentence_size + (1 if current_chunk else 0)
    current_chunk.append(sentence)
  if current_chunk:
      chunks.append(" ".join(current_chunk))

  return chunks


#EMBEDDING FUNCTION

client = chromadb.PersistentClient(path="chroma_db")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

collection = client.get_or_create_collection(
    name="documents_collection",
    embedding_function=sentence_transformer_ef
)

#INSERTING DATA INTO CHROMA DB
def process_document(file_path:str):
  try:
    content = read_pdf_file(file_path)

    chunks = split_text(content)

    #preparing metadata 
    file_name = os.path.basename(file_path)
    metadatas = [{"source":file_name, "chunk":i} for i in range(len(chunks))]
    ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

    return ids, chunks, metadatas

  except Exception as e:
    print(f"Error processing{file_path}:{str (e)}")
    return [],[],[]


def add_to_collection(collection,ids,texts,metadatas):

  if not texts:
    return 

  batch_size=50
  for i in range(0,len(texts),batch_size):
    end_idx = min (i+batch_size,len(texts))
    collection.add(
        ids = ids[i:end_idx],
        documents = texts[i:end_idx],
        metadatas = metadatas[i:end_idx]
    )
  

def process_and_add_documents(collection,folder_path:str):
  files = [os.path.join(folder_path,file) for file in os.listdir(folder_path) if file.endswith(".pdf")]
  for file_path in files:
    print(f"Processing {file_path}")
    ids,texts,metadatas = process_document(file_path)
    add_to_collection(collection,ids,texts,metadatas)
    print(f"Added {len(texts)} chunks to collection")


#SEMANTIC SEARCH

def semantic_search(collection, query: str, n_results: int = 2):
  return collection.query(
      query_texts=query,
      n_results=n_results,
  )


def get_context_with_sources(results):
  context = "\n\n".join(results['documents'][0])
  sources = [f"{meta['source']} (chunk{meta['chunk']})" for meta in results['metadatas'][0]]
  return context, sources


#LLM PROMPT + GENERATION

def get_prompt(query: str, context: str):
  prompt = f"""Based on the following context, please answer the following question. If the answer cannot be derived from the context, say "I cannot answer this based on the provided context."

  Context :
  {context}

  Question : {query}

  Answer:"""
  return prompt


def generate_response(query: str, context: str):
  prompt = get_prompt(query, context)
  response = groq_client.chat.completions.create(
      model="openai/gpt-oss-120b",
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt}
      ],
      temperature=0,
      max_tokens=1000
  )
  return response.choices[0].message.content


#RAG QUERY

def rag_query(collection, query: str, n_chunks: int = 2):
  results = semantic_search(collection, query)
  context, sources = get_context_with_sources(results)
  response = generate_response(query, context)
  return response, sources



#STREAMLIT UI

st.title("IITB Insti-Assist")
st.caption("Ask a question — answers come only from the ingested documents.")

# One-time insertion of documents into the collection
if collection.count() == 0:
    with st.spinner("Indexing documents..."):
        process_and_add_documents(collection, "documents")

query = st.text_input("Ask a question about IIT Bombay:")

if query:
    with st.spinner("Searching and generating answer..."):
        response, sources = rag_query(collection, query)

    st.write("**Answer:**", response)
    st.write("**Sources:**", sources)