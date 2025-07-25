from fastapi import FastAPI

app=FastAPI()

@app.get("/")
def read_root():
    return{"message":"helo world"}

@app.get("/hello/{name}")
def say_hello(name:str):
    return{"message":f"hello {name}"}

from pydantic import BaseModel

class query(BaseModel):
    userid: str
    message: str

@app.post("/chat/")
def chat(query: query):
    return {"message": f"User {query.userid} says: {query.message}"}




class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    answer: str

from together import Together
from dotenv import load_dotenv

load_dotenv()

client = Together()
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

@app.post("/chat", response_model=ChatResponse)
async def chat_with_llama(request: ChatRequest):
    """
    Receives a prompt from the user, sends it to the Llama 3 model,
    and returns the model's response.
    """
    
    # This is the core logic from your original script
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": request.prompt,
            }
        ]
    )
    
    # Extract the content from the first choice
    model_answer = response.choices[0].message.content

    return ChatResponse(answer=model_answer)