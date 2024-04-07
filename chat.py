import platform

if platform.system() != "Darwin":
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def make_chain(model_name: str, vector_store: Chroma) -> ConversationalRetrievalChain:
    """
    Creates a Chroma vector store and persists the documents to disk.

    :return: A chain with the specified model using the vector DB for retrieval.
    """


if __name__ == "__main__":
    load_dotenv()

    embedding = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name="jan-2024-economic",
        embedding_function=embedding,
        persist_directory="data/chroma",
    )

    chain = make_chain("gpt-3.5-turbo", vector_store)
    chat_history = []

    while True:
        print()
        question = input("Question: ")

        # Generate answer
        response = chain.invoke({"question": question, "chat_history": chat_history})

        # Parse answer
        answer = response["answer"]
        source = response["source_documents"]
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        # Display answer
        print("\n\nSources:\n")
        for document in source:
            print(f"Page: {document.metadata['page_number']}")
            print(f"Text chunk: {document.page_content[:160]}...\n")
        print(f"Answer: {answer}")
