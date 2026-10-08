"""AI 기본법 근거 기반 QA 백엔드 (FastAPI).

법률을 잘 모르는 사용자가 질문하면, 인공지능기본법 조문에서 근거를 찾아 쉬운 말로 설명하고 근거 조문 원문을 함께 돌려준다.
법령에 근거가 없는 질문은 지어내지 않고 답변을 거부한다.

실행 (프로젝트 루트에서):
    uv run uvicorn app.main:app --reload
    # 문서: http://127.0.0.1:8000/docs
"""

import logging
from functools import lru_cache

import openai
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from common.config import COLLECTION_NAME
from common.qdrant import get_qdrant_client
from rag.pipeline import RagPipeline, RagResult, cited_article

logger = logging.getLogger("app")

app = FastAPI(
    title="AI 기본법 QA",
    description="인공지능기본법 조문을 근거로 질문에 답하는 백엔드. 근거가 없으면 답변을 거부한다.",
    version="0.1.0",
)


# ----- 요청 / 응답 -----------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(description="인공지능기본법에 대한 질문", examples=["고영향 인공지능이란 무엇인가요?"])

    @field_validator("question")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = v.strip()
        if not 2 <= len(v) <= 500:
            raise ValueError("질문은 2자 이상 500자 이하로 입력해 주세요.")
        return v


class Source(BaseModel):
    citation: str = Field(description="인용 표기", examples=["인공지능기본법 제2조"])
    article: str = Field(description="조 라벨", examples=["제2조"])
    text: str = Field(description="근거 조문 원문 (조 전체)")


class AskResponse(BaseModel):
    question: str
    answer: str = Field(description="쉬운 말로 풀어쓴 답변. 거부한 경우 거부 안내문")
    refused: bool = Field(description="법령에 근거가 없어 답변을 거부했는가")
    refusal_reason: str | None = Field(description="거부 사유: low_relevance | not_answerable | no_verified_citation")
    citations: list[str] = Field(description="답변이 근거로 든 조항 (예: 제7조제8항)")
    sources: list[Source] = Field(description="답변이 근거로 든 조문 원문. 거부하면 비어 있다")


def to_response(result: RagResult) -> AskResponse:
    """파이프라인 결과를 응답으로 바꾼다. 근거 원문은 답변이 실제로 인용한 조만 담는다."""
    cited = {cited_article(c) for c in result.citations}
    sources = [] if result.refused else [
        Source(citation=c.citation, article=c.article, text=c.text) for c in result.contexts if c.article in cited
    ]
    return AskResponse(
        question=result.question,
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        citations=result.citations,
        sources=sources,
    )


# ----- 의존성 ---------------------------------------------------------------

@lru_cache
def get_pipeline() -> RagPipeline:
    """파이프라인은 한 번만 만든다 (클라이언트 생성만 하고 API 는 호출하지 않는다)."""
    return RagPipeline()


# ----- 엔드포인트 ------------------------------------------------------------

@app.get("/")
def root():
    return {"service": "AI 기본법 QA", "docs": "/docs", "ask": "POST /ask", "health": "GET /health"}


@app.get("/health")
def health():
    """Qdrant 연결과 법령 데이터 적재 여부를 확인한다 (LLM·임베딩 API 는 호출하지 않는다)."""
    try:
        client = get_qdrant_client()
        if not client.collection_exists(COLLECTION_NAME):
            raise HTTPException(503, f"컬렉션 '{COLLECTION_NAME}' 이 없습니다. 먼저 `python -m rag.pipeline ingest` 를 실행하세요.")
        points = client.count(COLLECTION_NAME).count
    except (ResponseHandlingException, UnexpectedResponse):
        raise HTTPException(503, "Qdrant 에 연결할 수 없습니다.") from None
    if points == 0:
        raise HTTPException(503, f"컬렉션 '{COLLECTION_NAME}' 이 비어 있습니다. 먼저 ingest 를 실행하세요.")
    return {"status": "ok", "collection": COLLECTION_NAME, "points": points}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> AskResponse:
    """질문에 인공지능기본법 조문을 근거로 답한다. 동기 함수라 FastAPI 가 스레드풀에서 실행한다(블로킹 호출이 서버를 막지 않는다)."""
    try:
        return to_response(pipeline.ask(req.question))
    except openai.RateLimitError:
        raise HTTPException(429, "요청이 많아 잠시 후 다시 시도해 주세요.") from None
    except openai.APIError:
        logger.exception("LLM/임베딩 API 오류")
        raise HTTPException(502, "답변 생성 서비스에 문제가 있습니다. 잠시 후 다시 시도해 주세요.") from None
    except (ResponseHandlingException, UnexpectedResponse):
        logger.exception("Qdrant 오류")
        raise HTTPException(503, "법령 검색 서비스에 연결할 수 없습니다.") from None
    except Exception:
        logger.exception("질문 처리 중 예상하지 못한 오류")
        raise HTTPException(500, "서버 내부 오류가 발생했습니다.") from None
