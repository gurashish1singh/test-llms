from __future__ import annotations

import os

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

DB_LOCATION = "./vector_db/chroma_langchain_db"


def vectorize(
    file_path: str = "realistic_restaurant_reviews.csv",
) -> VectorStoreRetriever:
    df = pd.read_csv(file_path)
    df.columns = list(map(str.lower, df.columns))
    add_documents: bool = not os.path.exists(DB_LOCATION)
    embedddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=DB_LOCATION,
        embedding_function=embedddings,
    )

    if add_documents:
        documents = []
        ids = []

        for row in df.itertuples():
            _id = row.Index
            document = Document(
                page_content=f"{row.title} {row.review}",
                metadata={"rating": row.rating, "date": row.date},
                id=str(_id),
            )
            ids.append(str(_id))
            documents.append(document)

        vector_store.add_documents(documents=documents, ids=ids)

    return vector_store.as_retriever(
        # N number of documents to pass in to llm
        search_kwargs={"k": 5},
    )
