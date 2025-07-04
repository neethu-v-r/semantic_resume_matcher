import faiss
import numpy as np

class ResumeIndex:
    def __init__(self):
        self.index = faiss.IndexFlatL2(384)
        self.resumes = []

    def add_resume(self, resume_text, embedding):
        self.index.add(np.array([embedding]))
        self.resumes.append(resume_text)

    def search(self, query_embedding, top_k=5):
        D, I = self.index.search(np.array([query_embedding]), top_k)
        return [(self.resumes[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
