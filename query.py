# # import faiss
# # import pickle
# # from sentence_transformers import SentenceTransformer
# # import os
# # from dotenv import load_dotenv
# # from openai import OpenAI


# # #1 load the same embedding model
# # model = SentenceTransformer("all-MiniLM-L6-v2")

# # #2 Load the saved FAISS index
# # index= faiss.read_index("vector_store/resume.index")

# # #load the original text chunks
# # with open("vector_store/chunks.pkl", "rb") as file:
# #     chunks= pickle.load(file)

# # def search_resume(question, top_k=3):
    
# #     ###convert the question into an embedding
# #     question_embedding= model.encode([question], convert_to_numpy=True)

# #     ##FAISS index was built using normalized vectors
# #     ##normalize the question vector too
# #     faiss.normalize_L2(question_embedding)

# #     ##find the most relevent chunks
# #     scores, indices =index.search(question_embedding, top_k)

# #     print("\nyour question:", question)
# #     print("\nRelevent information from your resume:\n")

# #     for rank, (score,idx) in enumerate(zip(scores[0], indices[0]), start=1):
# #         if idx == -1:
# #             continue

# #         print(f"Result{rank} | Similarity: {score:.4f}")
# #         print(chunks[idx])
# #         print("-" * 60)


# # ###Ask A Question
# # question= input("Ask something about your resume: ")
# # search_resume(question)




# """                              USING OPENAI_API                    """

# # load_dotenv()

# # api_key= os.getenv("OPENAI_API_KEY")

# # if not api_key:
# #     raise ValueError("OPENAI_API_KEY not found please check you .env file")

# # ##2 initialize OPENAI client 
# # client = OpenAI(api_key=api_key)

# # ##3Load the embedding model
# # model= SentenceTransformer("all-MiniLM-L6-v2")

# # ##4. load the saved FAISS index
# # index= faiss.read_index("vector_store/resume.index")

# # ##5. Load the original text chunks
# # with open("vector_store/chunks.pkl", "rb") as file:
# #     chunks= pickle.load(file)

# # def search_resume(question, top_k=3):
# #     ##Step1: Convert question into an embedding
# #     question_embedding = model.encode(
# #         [question],
# #         convert_to_numpy=True)

# #     ##Step 2. normalize the question embedding
# #     faiss.normalize_L2(question_embedding)

# #     ##Step 3. Retrieve relevent chunks from FAISS
# #     k= min(top_k, index.ntotal)
# #     scores, indices= index.search(question_embedding, k)

# #     retrieved_chunks= []

# #     for score, idx in zip(scores[0], indices[0]):
# #         if idx == -1:
# #             continue

# #         retrieved_chunks.append(chunks[idx])

# #     if not retrieved_chunks:
# #         print("no relevent information found in this resume")
# #         return 

# #     ##Step 4. combine retrieved chunks into context
# #     context= "\n\n".join(retrieved_chunks)

# #     ##step 5. create a prompt for LLM
# #     prompt= f"""
# # you are a resume assistanat.
# # Answer the user's question using only the information provided in the resume context. 
# # Instructions:
# # -Give a clear and direct answer.
# # Do not invent facts or experience.
# # If the answer is not available in the context,
# # say that the information is not available in the resume.

# # Resume context:
# # {context}

# # User Question:
# # {question}

# # Answer:
# # """
# #     # ##step6. send prompt to OpenAI
# #     # response= client.responses.create(
# #     #     model="gpt-4.1-mini",
# #     #     input=prompt)

# #     # ##Step 7. print the final generated answer
# #     # print("\nYour question:", question)
# #     # print("\nYour Answer:")
# #     # print(response.output_text)

# #     from openai import RateLimitError, APIError

# #     try:
# #         response = client.responses.create(
# #             model="gpt-4.1-mini",
# #             input=prompt
# #         )

# #         print("\nFinal Answer:")
# #         print(response.output_text)

# #     except RateLimitError as e:
# #         if getattr(e, "code", None) == "credit_balance_exhausted":
# #             print("\nYour OpenAI API credits are exhausted.")
# #             print("Check your API billing balance.")
# #         else:
# #             print("\nOpenAI API quota or rate limit error:", e)

# #     except APIError as e:
# #         print("\nOpenAI API error:", e)


# # ##Step8. Ask a question
# # if __name__ == "__main__":
# #     question = input("Ask something about your resume: ").strip()

# #     if question:
# #         search_resume(question)
# #     else:
# #         print("Please enter a question")

# # print("Query script reached the end")

# """                     USING  HUGGING FACE MODEL          """




# import faiss
# import pickle
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline


# # 2. Load the Hugging Face generation model
# # generator = pipeline(
# #     "text-generation",
# #     model="Qwen/Qwen2.5-1.5B-Instruct"
# # )
# generator = pipeline(
#     "text-generation",
#     model="Qwen/Qwen2.5-0.5B-Instruct",
#     device=-1
# )


# # 3. Load your embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# # 4. Load the FAISS index
# index = faiss.read_index("vector_store/resume.index")

# with open("vector_store/chunks.pkl", "rb") as file:
#     chunks = pickle.load(file)


# # 5. Your search function
# def search_resume(question, top_k=3):
#     question_embedding = embedding_model.encode(
#         [question],
#         convert_to_numpy=True
#     )

#     faiss.normalize_L2(question_embedding)

#     scores, indices = index.search(question_embedding, top_k)

#     # Collect relevant chunks
#     context = "\n\n".join(
#         chunks[i]
#         for i in indices[0]
#         if i >= 0
#     )

#     # Prepare prompt
#     prompt = f"""
#     You are a resume assistant.
#     Answer only using the provided resume context.
#     Do not invent information.

#     Resume Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """
#     print("Context retrieved successfully.")
#     print("Starting Hugging Face generation...")

#     print("\nGenerating answer...")

#     response = generator(
#         prompt,
#         # max_new_tokens=200,
#         max_new_tokens=50,
#         do_sample=False,
#         return_full_text=False
#     )

#     answer = response[0]["generated_text"].strip()
#     return answer

#     # print("\nFinal Answer:")
#     # print(answer.strip())


#     # # Generate answer using Hugging Face
#     # response = generator(
#     #     prompt,
#     #     max_new_tokens=200,
#     #     do_sample=False,
#     #     return_full_text=False
#     # )
#     # print("\nFinal Answer:")
#     # print(response[0]["generated_text"].strip())


# # # 6. Take user input 
# # question = input("Ask something about your resume: ")
# # search_resume(question)


            


# import os
# import pickle
# import os
# from dotenv import load_dotenv

# load_dotenv()

# api_key = os.environ.get("GROQ_API_KEY")
# import faiss
# from groq import Groq
# from sentence_transformers import SentenceTransformer


# # Initialize the Groq client using an environment variable
# api_key = os.environ.get("GROQ_API_KEY")

# if not api_key:
#     raise RuntimeError(
#         "GROQ_API_KEY is missing. Set it in your environment."
#     )

# client = Groq(api_key=api_key)



# import os
# import pickle
# import faiss

# from dotenv import load_dotenv
# from groq import Groq
# from sentence_transformers import SentenceTransformer

# load_dotenv()

# api_key = os.environ.get("GROQ_API_KEY")

# if not api_key:
#     raise RuntimeError("GROQ_API_KEY is missing from .env")

# client = Groq(api_key=api_key)


# # Load the embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# # Load the FAISS index
# index = faiss.read_index("vector_store/resume.index")


# # Load resume text chunks
# with open("vector_store/chunks.pkl", "rb") as file:
#     chunks = pickle.load(file)


# def search_resume(question, top_k=3):
#     """
#     Search the resume using FAISS and generate an answer using Groq.
#     """

#     # Step 1: Convert the question into an embedding
#     question_embedding = embedding_model.encode(
#         [question],
#         convert_to_numpy=True
#     )

#     # Step 2: Normalize the embedding
#     faiss.normalize_L2(question_embedding)

#     # Step 3: Search for relevant chunks
#     scores, indices = index.search(question_embedding, top_k)

#     # Step 4: Build context from the retrieved chunks
#     retrieved_chunks = [
#         chunks[i]
#         for i in indices[0]
#         if 0 <= i < len(chunks)
#     ]

#     if not retrieved_chunks:
#         return "I could not find relevant information in the resume."

#     context = "\n\n".join(retrieved_chunks)

#     # Step 5: Generate an answer using the hosted LLM
#     response = client.chat.completions.create(
#         # model="llama-3.3-70b-versatile",  ################################
#         model="openai/gpt-oss-120b",
#         messages=[
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a resume assistant. Answer only using "
#                     "the provided resume context. Do not invent "
#                     "personal details or experience. If the answer "
#                     "is unavailable in the context, say that the "
#                     "resume does not provide that information."
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     f"Resume Context:\n{context}\n\n"
#                     f"Question:\n{question}"
#                 )
#             }
#         ],
#         temperature=0.2,
#         max_completion_tokens=250
#     )

#     # Step 6: Return the generated answer
#     answer = response.choices[0].message.content

#     return answer.strip() if answer else "No answer was generated."




import os
import pickle

from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing.")

client = Groq(api_key=api_key)

# Load the resume chunks created by ingest.py
with open("vector_store/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)

if not isinstance(chunks, list) or not chunks:
    raise ValueError("chunks.pkl must contain a non-empty list.")

chunks = [str(chunk) for chunk in chunks if str(chunk).strip()]

if not chunks:
    raise ValueError("No valid resume text chunks found.")

# Build a lightweight text retrieval index
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

chunk_vectors = vectorizer.fit_transform(chunks)


def search_resume(question, top_k=3):

    question_lower = question.lower()

    # Expand common questions with related resume terms
    query_expansions = {
        "education": "education qualification degree BCA CGPA",
        "qualification": "education qualification degree BCA CGPA",
        "cgpa": "CGPA BCA Bachelor Computer Applications",
        "experience": "experience internship developer Python Django AI ML",
        "skills": "skills Python Django Flask machine learning",
        "projects": "projects developed applications",
    }

    expanded_question = question

    for keyword, related_terms in query_expansions.items():
        if keyword in question_lower:
            expanded_question += " " + related_terms

    question_vector = vectorizer.transform([expanded_question])

    scores = cosine_similarity(
        question_vector,
        chunk_vectors
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    retrieved_chunks = [
        chunks[i]
        for i in top_indices
        if scores[i] > 0
    ]

    if not retrieved_chunks:
        return (
            "I could not find relevant information in the resume. "
            "Try using keywords from your resume."
        )

    context = "\n\n".join(retrieved_chunks)

    response = client.chat.completions.create(
        # model="llama-3.3-70b-versatile",
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a resume assistant. Answer only using "
                    "the provided resume context. Do not invent details. "
                    "If information is unavailable, say so."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Resume Context:\n{context}\n\n"
                    f"Question:\n{question}"
                )
            }
        ],
        temperature=0.2,
        max_completion_tokens=250
    )

    answer = response.choices[0].message.content
    return answer.strip() if answer else "No answer was generated."