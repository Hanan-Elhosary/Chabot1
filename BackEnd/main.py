import together
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict
from rag_utils import prepare_documents, query_documents, generate_response
import os
import PyPDF2

load_dotenv()

api = FastAPI()

chat_histories: Dict[str, List[Dict[str, str]]] = {}
max_hist_keep = 15

class Query(BaseModel):
    user_id: str
    message: str

@api.post('/chat/')
def chat(query: Query):
    try:

        history = chat_histories.get(query.user_id, [])

        prompt = ""
        for turn in history:
            if turn["role"] == "User":
                prompt += f"<|user|>\n{turn['message']}\n"
            else:
                prompt += f"<|assistant|>\n{turn['message']}\n"

        prompt += f"<|user|>\n{query.message}\n<|assistant|>\n"

        response = together.Complete.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            prompt=prompt,
            max_tokens=256,
            stop=["<|user|>", "<|assistant|>"]
        )

        bot_response = response['choices'][0]['text'].strip().split("<|user|>")[0].strip()

        history.append({'role': 'User', 'message': query.message})
        history.append({'role': 'Chatbot', 'message': bot_response})
        chat_histories[query.user_id] = history[-max_hist_keep:]

        return {'response': bot_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
#RAG

def pdf_to_txt(pdf_path: str, output_dir: str):
    """
    Convert a PDF file to a .txt file and save it in output_dir.
    Returns the path of the created txt file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # اسم الملف بدون الامتداد
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}.txt")

    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(full_text)

    return txt_path
    
class RAGQuery(BaseModel):
    question: str

@api.post('/rag/')
def rag_chat(query: RAGQuery):
    try:
        pdf_files = [
            r"c:\Users\hnana\Downloads\microsoft-annual-report.pdf",
            r"c:c:\Users\hnana\Downloads\Hands-On Large Language Models Language Understanding and Generation (Jay Alammar, Maarten Grootendorst) .pdf"
        ]

        output_dir = r"c:\Users\hnana\Downloads\rag_texts"

        for pdf_path in pdf_files:
            pdf_to_txt(pdf_path, output_dir)

        prepare_documents(output_dir)

        relevant_chunks = query_documents(query.question)
        answer = generate_response(query.question, relevant_chunks)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

