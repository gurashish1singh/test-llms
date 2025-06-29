from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from vector import vectorize


def test_ollama():
    model = OllamaLLM(model="mistral")
    template = """
    You are an expert in answering questions about a pizze restaurant.
    Here are some relevant reviews: {reviews}

    Here is the question to answer: {question}
    """

    prompt = ChatPromptTemplate.from_template(template=template)
    # Pipe  creates a RunnableSequence where output of one callable is
    # passed in to the successive functions as an input.
    chain = prompt | model

    while True:
        print(f"\n{'-' * 20}\n")
        question = input("Ask your question (q to quit): ").strip()
        if question.lower() == "q":
            break

        reviews: list[Document] = vectorize().invoke(question)
        result = chain.invoke({"reviews": [reviews], "question": question})
        print(result)


if __name__ == "__main__":
    test_ollama()
