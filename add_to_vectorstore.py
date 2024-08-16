from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_content_extraction import pdf_extractor
import os
import re
import chromadb
from chromadb.config import Settings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import uuid
import sys

client=chromadb.HttpClient(host="localhost",port=8000,settings=Settings(anonymized_telemetry=False))

model_path = "/home/tuncube/.cache/gpt4all/all-MiniLM-L6-v2.gguf2.f16.gguf"

args = sys.argv
collection_name=str(args[1])

vectorstore = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function= GPT4AllEmbeddings(model_name=model_path,gpt4all_kwargs={'allow_download': False})
)
splitter = RecursiveCharacterTextSplitter(
    separators=["\n","\n\n"],
    chunk_size=512,
    chunk_overlap=32,
    length_function=len,
    is_separator_regex=False,
    keep_separator=False
)

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    
    # Removing Extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove unreadable characters
    text = ''.join(char for char in text if char.isprintable())

    # Remove bullets
    text = re.sub(r'•', ',', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Removing empty tokens
    tokens = [token for token in tokens if token]

    # Clean Text
    clean_text = ' '.join(tokens)

    return clean_text

def add_text(json_file,store):
    with open (json_file,'r') as f:
        dictionary=json.load(f)
    splitted=list()
    for i in dictionary:
        value=dictionary.get(i)
        if len(value.strip()) !=0:
            for a in splitter.split_text(value):
                splitted.append([re.sub(r'\d+', '', i).replace('.',' ').strip(), preprocess_text(a).strip()])
                

    doc_ids = [str(uuid.uuid4()) for _ in splitted]
    texts = [
        Document(page_content=s[1], metadata={"doc_id": doc_ids[i],"title" : s[0]})
        for i, s in enumerate(splitted)
    ]

    store.add_documents(texts)



folder="/home/tuncube/Documents/PFE/Ressources/JSON/"
folder_items=os.listdir(folder)
for file_name in folder_items:
    add_text(json_file=folder+file_name,store=vectorstore)