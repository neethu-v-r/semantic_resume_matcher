from fastapi import FastAPI, UploadFile, File
from embeddings import get_embedding
from faiss_index import ResumeIndex

app = FastAPI()
resume_index = ResumeIndex()

@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    content = await file.read()
    resume_text = content.decode("utf-8")
    emb = get_embedding(resume_text)
    resume_index.add_resume(resume_text, emb)
    return {"message": "Resume added."}

@app.post("/match/")
async def match(job_description: str):
    job_emb = get_embedding(job_description)
    matches = resume_index.search(job_emb, top_k=5)
    return {"matches": matches}
