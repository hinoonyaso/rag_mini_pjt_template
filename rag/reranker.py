"""Reranking: 검색된 child 후보를 LLM 으로 질문과의 관련도 순으로 다시 정렬한다.

별도 모델 설치 없이 기존 LLM(common.ai_model)을 쓰는 listwise 방식이다.
후보가 수십 개 이하인 이 프로젝트 규모를 전제로 하며, 호출이 실패하면 검색 순서를 그대로 쓴다.
(규모가 커지면 cross-encoder 로 교체할 수 있도록 rerank(query, hits, top_n) 인터페이스만 맞추면 된다.)
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from common.ai_model import get_llm_model
from rag.retriever import Hit

logger = logging.getLogger(__name__)


class _Score(BaseModel):
    id: int = Field(description="후보 번호")
    score: int = Field(ge=0, le=10, description="질문에 답하는 데 필요한 근거인 정도 (0~10)")


class _Ranking(BaseModel):
    scores: list[_Score] = Field(description="모든 후보에 대한 점수")


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 법령 검색 결과의 관련도를 채점하는 평가자다. "
            "질문에 답하는 데 직접 근거가 되는 조문일수록 높은 점수(10)를, "
            "단어만 겹치고 답과 무관하면 낮은 점수(0)를 준다. 모든 후보에 점수를 매겨라.",
        ),
        ("human", "질문: {question}\n\n후보:\n{candidates}"),
    ]
)


class LLMReranker:
    def __init__(self, llm=None, strict: bool = False):
        # strict=True 면 호출 실패 시 폴백하지 않고 예외를 낸다 (평가에서 실패를 숨기지 않기 위해)
        self.strict = strict
        llm = llm or get_llm_model(temperature=0, max_tokens=1024)
        # 이 프로젝트의 LLM 라우터는 function_calling 방식에서만 구조화 출력이 동작한다 (notebook/qdrant_test.ipynb).
        self.chain = _PROMPT | llm.with_structured_output(_Ranking, method="function_calling")

    def rerank(self, query: str, hits: list[Hit], top_n: int = 5) -> list[Hit]:
        if len(hits) <= 1:
            return hits[:top_n]
        candidates = "\n\n".join(f"[{i}] {h.citation}\n{h.text}" for i, h in enumerate(hits))
        try:
            ranking: _Ranking = self.chain.invoke({"question": query, "candidates": candidates})
        except Exception:
            if self.strict:
                raise
            logger.warning("rerank 호출 실패: 검색 순서를 그대로 사용한다.", exc_info=True)
            return hits[:top_n]

        scores = {s.id: s.score for s in ranking.scores if 0 <= s.id < len(hits)}
        # 점수가 높은 순, 같으면(또는 점수가 없으면) 원래 검색 순서 유지
        order = sorted(range(len(hits)), key=lambda i: (-scores.get(i, -1), i))
        for i, h in enumerate(hits):
            h.rerank_score = scores.get(i)
        return [hits[i] for i in order[:top_n]]
