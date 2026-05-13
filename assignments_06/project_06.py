from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path("./assignments_06/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

docs = SimpleDirectoryReader("./assignments_06/groundwork_docs").load_data()

print(f"Length of Documentts{len(docs)}")

for document in docs:
    print(document.metadata['file_name'])

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=3)

if query_engine:
    print("Index built successfully. Ready to answer questions.")

# questions = [
#     "What are Groundwork's hours on weekends?",
#     "Do you offer any dairy-free milk options?",
#     "How does the loyalty program work?",
#     "How did Groundwork Coffee get started?",
#     "Do you offer catering or wholesale orders?",
# ]

# for q in questions:
#     print(f"\nQ: {q}")
#     response = query_engine.query(q)
#     print("A:", response)
    
#     for node_with_score in response.source_nodes:
#         print(f"Node ID: {node_with_score.node.node_id}")
#         print(f"Similarity Score: {node_with_score.score:.4f}")
#         print(f"Text Snippet: {node_with_score.node.get_content()[:100]}...")
#         print("-" * 30)

#         #yes the agent sounds very confident, the part that is slighty surprising is the chat took on the tone of the writing as well

q2="what is the wifi password"

print(f"\nQ: {q2}")
response = query_engine.query(q2)
print("A:", response)

for node_with_score in response.source_nodes:
    print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
    print("-" * 30)

    #I expected it to be hard because answer is not in the documentation. It only says as the cashier for the wifi password
    #Wrong retrivial and I think the semantic meaning was off too
    #No it shouldnt just as sure and used similar snippets as well
    #I think the embedding process need to be better and add a system prompt to confirm the infromation is in the dopcuments and if it is not do not answer


    # This was significantly less code and only took about 5-6 lines, the value is that alot of the code is abstracted away and it is pretty easy to use out of the box and get running

    # This could be super helpful for a place like a library or reseach library if a user has questions about where certains types of books are or related topics

    # Halluncinations for LLM are almost impossible to prevent especially considering the possibility for user inputs
