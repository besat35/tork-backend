import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI(title="TORK Canlı Yapay Zeka Motoru")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # TORK'u tamamen ücretsiz, anahtarsız ve sınırsız canlı bir modele bağlıyoruz
        url = "https://openrouter.ai"
        headers = {
            "Authorization": "Bearer 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", # TORK'a özel taze canlı havuz anahtarı
            "Content-Type": "application/json"
        }
        
        # Ajanların karakterini belirleyen sistem talimatı
        system_instruction = (
            "Sen TORK AI uygulamasının arkasındaki akıllı çoklu ajan sistemisin. "
            "Kullanıcıya her zaman yardımcı, profesyonel, zeki ve samimi bir Türkçe ile yanıt ver. "
            "ChatGPT gibi davran, soruları net ve derinlemesine yanıtla."
        )

        data = {
            "model": "google/gemma-2-9b-it:free", # Google'ın en gelişmiş ücretsiz yapay zeka modeli
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": request.message}
            ]
        }
        
        res = requests.post(url, headers=headers, json=data).json()
        
        if 'choices' in res and len(res['choices']) > 0:
            response_text = res['choices']['message']['content']
        else:
            response_text = "TORK Ajanları şu an yoğun. Lütfen tekrar dener misiniz?"
            
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
