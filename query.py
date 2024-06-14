import os
import openai

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Set the environment variable OpenAI API key
os.environ["OPENAI_API_KEY"] = ""  # Key needs to be filled

# Read it
openai.api_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chromaDB"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():

    # Prompt the user to enter the query text
    query_text = input("Enter query: ")

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Initialize model
    model = ChatOpenAI()

    # Get the model's answer without context.
    direct_response_text = model.predict(query_text)
    print(f"Direct response: {direct_response_text}")

    print("\n\nLooking at the database...\n\n")

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=20)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    response_text = model.predict(prompt)

    sources = {doc.metadata.get("source", None) for doc, _score in results}
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
