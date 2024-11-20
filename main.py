from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, LLMChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import io
from PyPDF2 import PdfReader

# Çevresel değişkenleri yükle
load_dotenv()
HF_TOKEN='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'
# FastAPI uygulamasını başlat
app = FastAPI(title="RAG API", description="Retrieval Augmented Generation API")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production için frontend URL'inizi buraya ekleyin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RAGSystem:
    def __init__(self):
        # LLM modelini başlat
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=HF_TOKEN,
            task="text-generation",
            temperature=0.7,
            max_length=512
        )
        
        # Embedding modelini başlat
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Bellek ve vektör deposunu başlat
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.vector_store = None
        
        # Sohbet şablonunu oluştur
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Verilen bağlam bilgisini kullanarak soruyu yanıtla:
            Türkçe cevap ver. 

            Bağlam: {context}
            Soru: {question}
            
            Yanıt: """
        )

    def create_chain(self):
        """Farklı zincir türleri oluştur"""
        if not self.vector_store:
            raise ValueError("Önce bir doküman yüklemelisiniz!")

        # ConversationalRetrievalChain: Sohbet geçmişini hatırlayan ve bağlamı kullanan zincir
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={'k': 3}),
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
        
        # RetrievalQA: Basit soru-cevap zinciri
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True
        )
        
        return conv_chain

    def process_document(self, text: str):
        """Dokümanı işle ve vektör deposunu oluştur"""
        # Metni parçalara böl
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Vektör deposunu oluştur
        self.vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings
        )

# RAG sistemini başlat
rag_system = RAGSystem()

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[tuple]] = []

def extract_text_from_pdf(pdf_file_content: bytes) -> str:
    """PDF dosyasından metin çıkar"""
    pdf_reader = PdfReader(io.BytesIO(pdf_file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.get('/test')
def test():
    """Test endpoint"""
    return {"message": "Test successful"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Dosya yükle ve işle"""
    try:
        content = await file.read()
        
        # Dosya tipine göre metni çıkar
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(content)
        else:
            text = content.decode("utf-8")
        
        # Dokümanı işle
        rag_system.process_document(text)
        
        return {"message": "Dosya başarıyla yüklendi ve işlendi"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    """Sorguyu işle ve yanıt döndür"""
    try:
        chain = rag_system.create_chain()
        response = chain({
            "question": request.query,
            "chat_history": request.chat_history
        })
        
        return {
            "answer": response["answer"],
            "sources": [doc.page_content for doc in response["source_documents"]],
            "chat_history": response.get("chat_history", [])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
