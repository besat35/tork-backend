import os
from dotenv import load_dotenv

# .env dosyasındaki değişkenleri sisteme yükler
load_dotenv()
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="TORK Canlı Yapay Zeka Motoru")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct:free"

@app.get("/")
def health():
    return {"status": "ok", "service": "TORK AI"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY Render Environment Variables bölümünde tanımlı değil."
        )

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Mesaj boş olamaz.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://torkyzeka.com.tr",
        "X-Title": "TORK Yapay Zeka",
    }

    system_instruction = (
        "Sen TORK AI uygulamasının arkasındaki akıllı asistansın. "
        "Kullanıcıya yardımcı, profesyonel, zeki ve samimi bir Türkçe ile yanıt ver. "
        "Soruları net ve anlaşılır biçimde yanıtla."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": message},
        ],
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"OpenRouter bağlantı hatası: {exc}")

    # Never call .json() blindly: OpenRouter may return non-JSON on an upstream error.
    try:
        data = response.json()
    except ValueError:
        preview = response.text[:300].replace("\n", " ")
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter geçerli JSON döndürmedi (HTTP {response.status_code}): {preview}"
        )

    if response.status_code >= 400:
        error = data.get("error", data) if isinstance(data, dict) else data
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter API hatası (HTTP {response.status_code}): {error}"
        )

    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter yanıtında choices bulunamadı: {data}"
        )

    content = choices[0].get("message", {}).get("content")
    if not content:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter yanıtında mesaj içeriği bulunamadı: {data}"
        )

    return {"response": content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
