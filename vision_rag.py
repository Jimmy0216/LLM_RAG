from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import base64
from byaldi import RAGMultiModalModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import uvicorn
import pdfplumber
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# 환경 변수 로드
load_dotenv()

app = FastAPI()

# RAG 모델 초기화 (CPU 버전)
try:
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1, device="cpu")
    logger.info("RAG 모델 초기화 성공")
except Exception as e:
    logger.error(f"RAG 모델 초기화 실패: {str(e)}", exc_info=True)
    raise

import pdfplumber
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def index_document(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        
        # 텍스트를 새 PDF로 변환
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        textobject = c.beginText(40, 750)
        for line in text.split('\n'):
            textobject.textLine(line)
        c.drawText(textobject)
        c.save()
        pdf_buffer.seek(0)
        
        # 메모리 내 PDF 파일을 사용하여 인덱싱
        RAG.index(
            input_path=pdf_buffer,
            index_name="korean_doc",
            store_collection_with_index=True,
            overwrite=True
        )

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

def mm_chat(img_base64, prompt):
    # GPT-4o 모델 사용
    chat = ChatOpenAI(model="gpt-4o")  # 모델명 유지
    msg = chat.invoke(
        [
            SystemMessage(content="You are a helpful assistant that can answer questions about images."),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    }
                ]
            )
        ],
        max_tokens=1024
    )
    return msg.content

import pdfplumber
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def index_document(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        
        # 텍스트를 새 PDF로 변환
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        textobject = c.beginText(40, 750)
        for line in text.split('\n'):
            textobject.textLine(line)
        c.drawText(textobject)
        c.save()
        pdf_buffer.seek(0)
        
        # 메모리 내 PDF 파일을 사용하여 인덱싱
        RAG.index(
            input_path=pdf_buffer,
            index_name="korean_doc",
            store_collection_with_index=True,
            overwrite=True
        )

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        index_document(temp_file_path)
        return JSONResponse(content={"message": "문서 인덱싱 완료!"}, status_code=200)
    except Exception as e:
        logger.error(f"인덱싱 중 오류 발생: {str(e)}", exc_info=True)
        return JSONResponse(content={"message": f"인덱싱 중 오류 발생: {str(e)}"}, status_code=500)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/query")
async def query(question: str = Form(...)):
    results = RAG.search(question, k=1)
    if results:
        image_base64 = results[0].base64
        answer = mm_chat(image_base64, question)
        return JSONResponse(content={
            "answer": answer,
            "image": image_base64
        }, status_code=200)
    else:
        return JSONResponse(content={"message": "관련 정보를 찾을 수 없습니다."}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)