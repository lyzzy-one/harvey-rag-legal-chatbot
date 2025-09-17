from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

try:
    from openai import OpenAI  # openai>=1.x
except Exception:
    OpenAI = None


class PredictRequest(BaseModel):
    query: str
    top_k: int = 5


class PredictResponseItem(BaseModel):
    text: str
    score: float


class PredictResponse(BaseModel):
    predicted_label: Optional[str] = None
    items: List[PredictResponseItem]


def load_resources(prefix: str = "lbox_casename"):
    df = pd.read_parquet(f"{prefix}_embeddings.parquet")
    index = faiss.read_index(f"{prefix}.faiss")
    # 간단한 레이블: casetype을 사용할 수 있으면 parquet에 포함하도록 커스텀 가능
    labels = None
    if "casetype" in df.columns:
        labels = df["casetype"].tolist()
    return df, index, labels


app = FastAPI(title="Legal Case Outcome Predictor (kNN)")

# CORS (로컬 개발 기본 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일: /ui 경로로 제공 (API 라우트와 충돌 방지)
if os.path.isdir("static"):
    app.mount("/ui", StaticFiles(directory="static", html=True), name="static")


@app.on_event("startup")
def on_startup():
    global DF, INDEX, MODEL, LABELS
    DF, INDEX, LABELS = load_resources("lbox_casename")
    MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    query_vec = MODEL.encode([req.query], normalize_embeddings=True).astype(np.float32)
    scores, ids = INDEX.search(query_vec, req.top_k)
    top_items: List[PredictResponseItem] = []
    for i, s in zip(ids[0], scores[0]):
        text = DF.iloc[i]["text"]
        top_items.append(PredictResponseItem(text=text, score=float(s)))

    # 간단한 레이블 투표(있을 경우): LABELS가 None이면 생략
    predicted_label = None
    if LABELS is not None:
        candidates = [LABELS[i] for i in ids[0]]
        # 최빈값
        if candidates:
            predicted_label = max(set(candidates), key=candidates.count)

    return PredictResponse(predicted_label=predicted_label, items=top_items)


@app.get("/")
def root():
    return {"ok": True}


class RagRequest(BaseModel):
    query: str
    top_k: int = 5
    max_tokens: int = 400


class RagResponse(BaseModel):
    answer: str
    items: List[PredictResponseItem]


def _ensure_openai_client() -> Optional["OpenAI"]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


@app.post("/predict_rag", response_model=RagResponse)
def predict_rag(req: RagRequest):
    # 1) 검색
    query_vec = MODEL.encode([req.query], normalize_embeddings=True).astype(np.float32)
    scores, ids = INDEX.search(query_vec, req.top_k)
    top_items: List[PredictResponseItem] = []
    contexts: List[str] = []
    for i, s in zip(ids[0], scores[0]):
        text = DF.iloc[i]["text"]
        top_items.append(PredictResponseItem(text=text, score=float(s)))
        contexts.append(text)

    # 2) LLM 요약 (키 없으면 fallback)
    client = _ensure_openai_client()
    if client is None:
        # 간단한 규칙 기반 요약
        snippet = "\n\n".join(t[:400] for t in contexts[:3])
        answer = (
            "[LLM 미사용 요약] 유사 판례 근거로 본 예상 판단:\n" + snippet
        )
        return RagResponse(answer=answer, items=top_items)

    # 프롬프트 구성
    system_prompt = (
        "당신은 한국 법률 도메인의 보조 판사입니다. 제공된 유사 판례들을 근거로, "
        "질문된 사건에 대해 예상되는 판단과 핵심 근거 조항/논리를 간결히 정리해 주세요. "
        "법률 자문이 아닌 정보 제공의 범위를 지키고, 확실하지 않은 부분은 추정임을 명시하세요."
    )
    context_block = "\n\n".join(f"[사례 {i+1}]\n{c}" for i, c in enumerate(contexts))
    user_prompt = (
        f"질문:\n{req.query}\n\n"
        f"근거 판례들:\n{context_block}\n\n"
        "요청: 위 판례를 참고하여 사건의 예상 판단을 5~8문장으로 요약하고, "
        "핵심 근거(요건 충족 여부, 법조문 수준)를 bullet로 제시하세요."
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=req.max_tokens,
    )
    answer = completion.choices[0].message.content.strip()
    return RagResponse(answer=answer, items=top_items)


