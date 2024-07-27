import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pydantic import BaseModel
from dotenv import load_dotenv
import faiss
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGO_URI")
DB_NAME = "langchain_db"
COLLECTION_NAME = "test"

mongo_client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
MONGODB_COLLECTION = mongo_client[DB_NAME][COLLECTION_NAME]

class QueryRequest(BaseModel):
    query: str

class BookResponse(BaseModel):
    titles: list

class LessonPlanRequest(BaseModel):
    topic: str
    subtopic: str
    session1Duration: int
    session2Duration: int

@app.post("/query_books")
async def query_books(request: QueryRequest):
    query = request.query
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    documents = list(MONGODB_COLLECTION.find({}))
    if not documents:
        raise HTTPException(status_code=404, detail="No documents found in the database.")

    texts = [doc["text"] for doc in documents]
    vectors = [doc["embedding"] for doc in documents]

    if not vectors:
        raise HTTPException(status_code=500, detail="No vectors found in the documents.")

    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors).astype("float32"))

    query_vector = embeddings.embed_query(query)

    D, I = index.search(np.array([query_vector]).astype("float32"), k=5)
    results = [texts[i] for i in I[0]]

    return {"results": results}

@app.post("/upload_book")
async def upload_book(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(file_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

        for doc in docs:
            embedding = embeddings.embed_documents([doc.page_content])[0]
            MONGODB_COLLECTION.insert_one({
                "text": doc.page_content,
                "embedding": embedding,
                "title": file.filename
            })

        return {"message": "Book uploaded and processed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process the book: {str(e)}")

@app.get("/list_books", response_model=BookResponse)
async def list_books():
    documents = list(MONGODB_COLLECTION.find({}))
    titles = list(set([doc.get("title", "Unknown") for doc in documents]))
    return {"titles": titles}

@app.post("/generate_lesson_plan")
async def generate_lesson_plan(request: LessonPlanRequest):
    topic = request.topic
    subtopic = request.subtopic
    session1_duration = request.session1Duration
    session2_duration = request.session2Duration

    query = f"{topic} {subtopic}"
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    documents = list(MONGODB_COLLECTION.find({}))
    if not documents:
        raise HTTPException(status_code=404, detail="No documents found in the database.")

    texts = [doc["text"] for doc in documents]
    vectors = [doc["embedding"] for doc in documents]
    titles = [doc["title"] for doc in documents]

    if not vectors:
        raise HTTPException(status_code=500, detail="No vectors found in the documents.")

    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors).astype("float32"))

    query_vector = embeddings.embed_query(query)

    D, I = index.search(np.array([query_vector]).astype("float32"), k=5)
    results = [texts[i] for i in I[0]]
    references = [titles[i] for i in I[0]]

    detailed_descriptions = []
    for result in results:
        detailed_descriptions.append(requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'gpt-4',
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': f'Provide a detailed explanation and additional information about the following topic: {result}'}
                ],
                'max_tokens': 1024,
                'temperature': 0.7
            }
        ).json()['choices'][0]['message']['content'])

    openai_response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
            'Content-Type': 'application/json'
        },
        json={
            'model': 'gpt-4',
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': f'Create a detailed KTP (knowledge transfer plan) for one week on the subject of {topic} with a focus on the theme of {subtopic}. The KTP should be divided into two sessions, each focusing on developing understanding of {subtopic}. Each session should be tailored to the given duration: {session1_duration} minutes for the first session and {session2_duration} minutes for the second session. Each session should include multiple segments with varying durations that sum up to the total session time. Provide a detailed breakdown of each segment including theme, duration, and description. Use the following references: {", ".join(references)}. Format the response as follows: Session 1: Theme: [theme] Duration: [duration] minutes Description: [description] Methodology: [methodology] ... Session 2: Theme: [theme] Duration: [duration] minutes Description: [description] Methodology: [methodology] ... References: [reference]'}
            ],
            'max_tokens': 2048,
            'temperature': 0.7
        }
    )

    if openai_response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to generate lesson plan")

    ktp_content = openai_response.json()['choices'][0]['message']['content']
    return {"lesson_plan": ktp_content, "references": references, "detailed_descriptions": detailed_descriptions}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
