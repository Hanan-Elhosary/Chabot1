import together
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict

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
            max_tokens=1024,
            stop=["<|user|>", "<|assistant|>"]
        )

        bot_response = response['choices'][0]['text'].strip().split("<|user|>")[0].strip()

        history.append({'role': 'User', 'message': query.message})
        history.append({'role': 'Chatbot', 'message': bot_response})
        chat_histories[query.user_id] = history[-max_hist_keep:]

        return {'response': bot_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
