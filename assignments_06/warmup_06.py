# --- RAG Concepts ---
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

 # Concepts Q1
  #Scenario A:
#RAG would be best suited for this use case because it would include parsing documents, if it was Fine tuning that would be training the model on content that is soon to be outdataed

  #Scenario B:
#Prompt engineering since there are already lots of examples and a clear standard they want the outcome to be written in this is best suited for multiple shot prompt engineering
  #Scenario C:
#Prompt engineering would also be optimal for this since it is only once the time and energy that would be needed for RAG or Finet tuning would be disporportionate.

 # Concepts Q2
 #halluncination are dangerous because in many cases the end user is using the LLM as the source of truth, so theses hallunciation can change the behavior and outcome of the end user unintentionally.
 #There have been halluncinations with the chat bot I work with where it has told customers about refunds being processed and they never happened. This lead to confusion an dcustomers to get updet with the business.

 #In general, with human confidence means a lot. responses that are very matter of fact and devoid of emotion are often prioritized over more honest responses or responses that exist somewhere in the gray.

  # Concepts Q3
steps = [
    "Extract text from source documents",
    "Split text into chunks",
    "Convert text chunks into embeddings",
    "Receive the user's query",
    "Embed the user's query",
    "Retrieve the most relevant chunks",
    "Inject retrieved chunks into the prompt",
    "Generate a response from the LLM",
]
#Keyword RAG
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


query = "What are your hours on the weekend?"
query2 = "Do you have anything without caffeine?"
query3 = "How do I sign up for rewards?"



documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

response= simple_keyword_retrieval(query3, documents, verbose=True)

print(response)

# Keyword Q1
#loyalty document was chisen because it was one of the 2 documents that had the would your in which was in the user's orginal query. Even though semantically gours document was more relevant

# Keyword Q2
#none of the documents were chosen because none of the keywords matched. It did not get it right because the preview of the keyword search is limited. Semantic RAG would be much better than a keyord search 

# Keyword Q3
#my predication is that it will not match with any of them since none of the sematic keyword align with what is inside of the users query
#although the  loyalty document matched the query, it was not choosen and no I am not surprised by the repsone--there are no overlapping key words 


#--Semantic RAG Concepts--

#Semantic Q1
# vector embedding is the process of turning tokens into vectors and vectors are just mathmatical representations of the meaning of words and their proximity in meaning to other words.
#TDLR, vector embedding is turning natural languages into language(math) that computers can understand

# the chunk that is more relevant is the 0.85, because the closer it is to 1 represents how similar in meaning the input and the chunk are.

#Sematic search find the most relevant chucking using vector embedding which is taking the semantic meaning of words a allows the compter to compare to the semantic meaning of the chunk


#Semantic Q2
# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | semantic meaning of the word|
# | What is retrieved?         | Full document                     | chucnks      |
# | Can it handle synonyms?    | No                                | yes          |
# | Storage format             | Plain text dictionary             | database     |
# | Relevance score            | Number of overlapping keywords    | cosign similarity|


#--LlamaIndex--
docs = SimpleDirectoryReader("./assignments_06/brightleaf_pdfs").load_data()

index = VectorStoreIndex.from_documents(docs)
print(type(index._vector_store).__name__)

query_engine1 = index.as_query_engine(similarity_top_k=1)
# query_engine3 = index.as_query_engine(similarity_top_k=3)
# query_engine5 = index.as_query_engine(similarity_top_k=5)

# questions = [
#     "What employee benefits does BrightLeaf offer?",
#     "What are BrightLeaf's security policies?",
# ]
questions=[
    "how is solar done at bright solar and how is it run?"
]

for q in questions:
    print(f"\nQ: {q}")
    response1 = query_engine1.query(q)
    print("A:", response1)
    # response3 = query_engine3.query(q)
    # print("A:", response3)
    # response5 = query_engine5.query(q)
    # print("A:", response5)
    
    for node_with_score in response1.source_nodes:
        print("RESPONSE 1")
        print("-" * 30)
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)
    # for node_with_score in response3.source_nodes:
    #     print("RESPONSE 3")
    #     print("-" * 30)
    #     print(f"Node ID: {node_with_score.node.node_id}")
    #     print(f"Similarity Score: {node_with_score.score:.4f}")
    #     print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    #     print("-" * 30)
    # for node_with_score in response5.source_nodes:
    #     print("RESPONSE 5")
    #     print("-" * 30)
    #     print(f"Node ID: {node_with_score.node.node_id}")
    #     print(f"Similarity Score: {node_with_score.score:.4f}")
    #     print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    #     print("-" * 30)

#LlamaIndex Q1
# the retrieved chunks are very relvant to the user's queries
#the tone of the model is very seure and moatter of fact regardless of correctness
#oth the second and the third chunks seem largely unrelated from both answers but their scores are relatively high

#LlamaIndex Q2
#There were was no difference between the score and the top K for k of 1,3,5

#LlamaIndex Q3
# I expected a models response to be better and the way that it could be better is for the system to ask clarification
#if the query is clear or asking the user if that answered the question.


def llama_index_pipeline(k=3, questions=""):
    docs = SimpleDirectoryReader("./assignments_06/brightleaf_pdfs").load_data()

    index = VectorStoreIndex.from_documents(docs)
    print(type(index._vector_store).__name__)

    query_engine = index.as_query_engine(similarity_top_k=k)


    for q in questions:
        print(f"\nQ: {q}")
        response = query_engine.query(q)
        print("A:", response)

    for node_with_score in response.source_nodes:
        print("-" * 30)
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

#LlamaIndex Q4
def response_eval(question, k=3):
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

    # Define evaluator
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    relevancy_evaluator = RelevancyEvaluator(llm=llm)

    query_engine = index.as_query_engine(similarity_top_k=k)
    response = query_engine.query(question)

    # Evaluate faithfulness and relevancy
    faithfulness_result = faithfulness_evaluator.evaluate_response(query=question, response=response)
    print("Faithfulness Evaluation: " + str(faithfulness_result.score))

    relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
    print("Relevancy Result: " + str(relevancy_result.score))
q = "What employee benefits does BrightLeaf offer?"
q2="how is solar done at bright solar and how is it run?"
response_eval(q)

response_eval(q2)

#faithfulness and relevance is a binary, either yes or no. So 1 means that the response is both relevant and faithful
#relevancy means the responses is semantically matches the user's query and faithfulness is if the derived content is ffrom the content or hallucinations
#for my vague question the relevancy was zero, but that is mostly related to my query lacking very semantic meaning. Lots of non descript and off topic words

#accurancy is hard to judge and very subjective. LLM as a judge allows for a scalable wa to check for relevance and faithfulness
