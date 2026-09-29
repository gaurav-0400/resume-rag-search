import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from openai import OpenAI


# #1 load the same embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# #2 Load the saved FAISS index
# index= faiss.read_index("vector_store/resume.index")

# #load the original text chunks
# with open("vector_store/chunks.pkl", "rb") as file:
#     chunks= pickle.load(file)

# def search_resume(question, top_k=3):
    
#     ###convert the question into an embedding
#     question_embedding= model.encode([question], convert_to_numpy=True)

#     ##FAISS index was built using normalized vectors
#     ##normalize the question vector too
#     faiss.normalize_L2(question_embedding)

#     ##find the most relevent chunks
#     scores, indices =index.search(question_embedding, top_k)

#     print("\nyour question:", question)
#     print("\nRelevent information from your resume:\n")

#     for rank, (score,idx) in enumerate(zip(scores[0], indices[0]), start=1):
#         if idx == -1:
#             continue

#         print(f"Result{rank} | Similarity: {score:.4f}")
#         print(chunks[idx])
#         print("-" * 60)


# ###Ask A Question
# question= input("Ask something about your resume: ")
# search_resume(question)





load_dotenv()

api_key= os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found please check you .env file")

##2 initialize OPENAI client 
client = OpenAI(api_key=api_key)

##3Load the embedding model
model= SentenceTransformer("all-MiniLM-L6-v2")

##4. load the saved FAISS index
index= faiss.read_index("vector_store/resume.index")

##5. Load the original text chunks
with open("vector_store/chunks.pkl", "rb") as file:
    chunks= pickle.load(file)

def search_resume(question, top_k=3):
    ##Step1: Convert question into an embedding
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True)

    ##Step 2. normalize the question embedding
    faiss.normalize_L2(question_embedding)

    ##Step 3. Retrieve relevent chunks from FAISS
    k= min(top_k, index.ntotal)
    scores, indices= index.search(question_embedding, k)

    retrieved_chunks= []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        retrieved_chunks.append(chunks[idx])

    if not retrieved_chunks:
        print("no relevent information found in this resume")
        return 

    ##Step 4. combine retrieved chunks into context
    context= "\n\n".join(retrieved_chunks)

    ##step 5. create a prompt for LLM
    prompt= f"""
you are a resume assistanat.
Answer the user's question using only the information provided in the resume context. 
Instructions:
-Give a clear and direct answer.
Do not invent facts or experience.
If the answer is not available in the context,
say that the information is not available in the resume.

Resume context:
{context}

User Question:
{question}

Answer:
"""
    # ##step6. send prompt to OpenAI
    # response= client.responses.create(
    #     model="gpt-4.1-mini",
    #     input=prompt)

    # ##Step 7. print the final generated answer
    # print("\nYour question:", question)
    # print("\nYour Answer:")
    # print(response.output_text)
    from openai import RateLimitError, APIError

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        print("\nFinal Answer:")
        print(response.output_text)

    except RateLimitError as e:
        if getattr(e, "code", None) == "credit_balance_exhausted":
            print("\nYour OpenAI API credits are exhausted.")
            print("Check your API billing balance.")
        else:
            print("\nOpenAI API quota or rate limit error:", e)

    except APIError as e:
        print("\nOpenAI API error:", e)


##Step8. Ask a question
if __name__ == "__main__":
    question = input("Ask something about your resume: ").strip()

    if question:
        search_resume(question)
    else:
        print("Please enter a question")

print("Query script reached the end")
            


