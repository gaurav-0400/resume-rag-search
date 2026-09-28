import faiss
import pickle
from sentence_transformers import SentenceTransformer


#1 load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

#2 Load the saved FAISS index
index= faiss.read_index("vector_store/resume.index")

#load the original text chunks
with open("vector_store/chunks.pkl", "rb") as file:
    chunks= pickle.load(file)

def search_resume(question, top_k=3):
    
    ###convert the question into an embedding
    question_embedding= model.encode([question], convert_to_numpy=True)

    ##FAISS index was built using normalized vectors
    ##normalize the question vector too
    faiss.normalize_L2(question_embedding)

    ##find the most relevent chunks
    scores, indices =index.search(question_embedding, top_k)

    print("\nyour question:", question)
    print("\nRelevent information from your resume:\n")

    for rank, (score,idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:
            continue

        print(f"Result{rank} | Similarity: {score:.4f}")
        print(chunks[idx])
        print("-" * 60)


###Ask A Question
question= input("Ask something about your resume: ")
search_resume(question)


