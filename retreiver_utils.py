from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as faiss_vdb
from langchain_community.embeddings import HuggingFaceEmbeddings as TextEmbeddings
from configs import (
    CHUNKS_SIZE,
    CHUNKS_OVERLAP_SIZE,
    DB_NAME,
    DEVICE,
    EMBEDDER_MODEL
)
from pprint import pprint

class DBretriever:

    def __init__(self, director_or_path: str):

        text_loader_kwargs = {"autodetect_encoding": True}

        data_loader = DirectoryLoader(
            director_or_path,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs=text_loader_kwargs,
        )
        text_chunker = RecursiveCharacterTextSplitter(
            chunk_size=CHUNKS_SIZE, chunk_overlap=CHUNKS_OVERLAP_SIZE
        )

        documents = data_loader.load()
        chunks = text_chunker.split_documents(documents)

        pprint("preparing text embedders...")
        embedder = TextEmbeddings(
            model_name=EMBEDDER_MODEL,
            model_kwargs={"device": DEVICE},
        )

        pprint("saving database...")
        try:
            # do not recreate the database every time, just load it
            db_hander = faiss_vdb.load_local(DB_NAME)
        except:
            db_handler = faiss_vdb.from_documents(chunks, embedder)
            db_handler.save_local(DB_NAME)

        pprint("preparing similarity retriever...")
        # Tell the database that we need it to work as retriever. It uses similarity measures for high dimensional vectors, and it returns top 3 results. These results will be given to the LLM to help it answer the user's questions.
        self.db_retriever = db_handler.as_retriever(
            search_kwargs={"k": 3, "search_type": "similarity"}
        )
        
        
        
    
