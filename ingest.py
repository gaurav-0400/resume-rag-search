# import fitz
# pdf_path= "resume.pdf"
# doc= fitz.open(pdf_path)
# text = ""
# for page_number, page in enumerate(doc, start=1):
#     page_text = page.get_text()
#     text +=page_text
#     print(f"Page:{page_number}")
#     print(page_text)
# print("Total character:" , len(text))



import pymupdf
# STEP 1: Read the PDF
pdf_path = "resume.pdf"

doc = pymupdf.open(pdf_path)

text = ""

for page_number, page in enumerate(doc, start=1):
    page_text = page.get_text("text", sort=True)
    text += page_text + "\n"

print(text)
print("\nTotal characters:", len(text))

doc.close()


# STEP 2: Create chunks
def create_chunks(text,chunk_size=500, overlap=80):
    chunks=[]
    start=0

    while start < len(text):
        end= start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

            start+=chunk_size - overlap

    return chunks


# STEP 3: Call the function

chunks= create_chunks(text)




#########################################################
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert chunk list into text input
chunk_texts = chunks

# Generate embeddings for all chunks
embeddings = model.encode(
    chunk_texts,
    convert_to_numpy=True
)

# Convert embeddings to float32
embeddings = np.asarray(embeddings, dtype="float32")

# Display results
print("\nTotal chunks:", len(chunks))
print("Embeddings shape:", embeddings.shape)

for i, embedding in enumerate(embeddings):
    print(f"\nChunk {i + 1}")
    print("Vector:", embedding)
    print("Dimensions:", len(embedding))


###############################################################    

# STEP 4: Display the chunks
print("\nTotal chunks created:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n---CHUNK {i}---")
    print(chunk)
    print("characters:", len(chunk))



#######FAISS vector database
import faiss
import pickle
import os 

##creating a folder to store my vector database files
os.makedirs("vector_store", exist_ok=True)

##normalize vectors for cosine-similarity-style search

faiss.normalize_L2(embeddings)

##creating a FAISS index
dimension= embeddings.shape[1]
index= faiss.IndexFlatIP(dimension)


##add my embeddings to FAISS
index.add(embeddings)

##saving the FAISS index
faiss.write_index(index, "vector_store/resume.index")


##save the original text chunks
with open("vector_store/chunks.pkl","wb") as file:
    pickle.dump(chunks, file)

print("Total chunks:", len(chunks))
print("embedding shape:", embeddings.shape)
print("vdctor stored in FAISS:", index.ntotal)
print("FAISS and chunks saved successfully")









