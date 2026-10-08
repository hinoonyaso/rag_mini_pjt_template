"""전체 RAG 파이프라인.

적재(ingest): 법령 API(XML) → Canonical Legal JSON → Parent-Child 청킹 → Embedding → Qdrant
질의(ask)   : Hybrid Retrieval → Reranking → 조(parent) 단위 컨텍스트 → LLM → 근거 조문 기반 답변

CLI (프로젝트 루트에서):
    uv run python -m rag.pipeline parse [--refresh]  # API(XML) → Canonical JSON 만 생성
    uv run python -m rag.pipeline ingest [--refresh] [--recreate]
    uv run python -m rag.pipeline ask "고영향 인공지능이란?"
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from common.ai_model import get_llm_model
from common.config import CANONICAL_JSON_PATH, COLLECTION_NAME, RAG_MIN_RELEVANCE
from rag.chunker import DEFAULT_MAX_CHARS, chunk_law
from rag.loader import load_law, save_canonical
from rag.reranker import LLMReranker
from rag.retriever import HybridRetriever, ParentContext, group_by_parent
from rag.vectorstore import index_chunks

_RE_ARTICLE_REF = re.compile(r"제\d+조(?:의\d+)?")

def cited_article(cite: str) -> str | None:
    """인용 표기("제7조제8항", "부칙 제1조")에서 조 라벨("제7조", "부칙")을 뽑는다."""
    cite = cite.strip()
    if cite.startswith("부칙"):
        return "부칙"
    m = _RE_ARTICLE_REF.match(cite)
    return m.group(0) if m else None


REFUSAL_MESSAGE = (
    "제공된 인공지능기본법 조문에서 이 질문의 근거를 찾을 수 없어 답변드릴 수 없습니다. "
    "법령에 규정된 내용으로 다시 질문해 주세요."
)


# ----- 적재 ---------------------------------------------------------------

def build_canonical(refresh: bool = False, out_path: str | Path = CANONICAL_JSON_PATH) -> dict:
    """법령 XML(캐시 또는 API)을 파싱해 Canonical Legal JSON 을 만들고 파일로 저장한다."""
    doc = load_law(refresh=refresh)
    save_canonical(doc, out_path)
    return doc


def ingest(
    refresh: bool = False,
    recreate: bool = False,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict:
    doc = build_canonical(refresh)
    chunks = chunk_law(doc, max_chars=max_chars)
    stored = index_chunks(chunks, recreate=recreate)
    return {"articles": len(doc["articles"]), "chunks": len(chunks), "stored": stored}


# ----- 질의 ---------------------------------------------------------------

class _RagAnswer(BaseModel):
    answer: str = Field(description="질문에 대한 답변. 근거 조문을 본문에서 함께 언급한다.")
    cited_articles: list[str] = Field(
        description="답변의 근거가 된 조항 표기 목록 (예: ['제2조제4호', '제6조제2항'])"
    )
    is_answerable: bool = Field(description="제공된 조문만으로 답변할 수 있었는지 여부")


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 제공된 법령 조문만을 근거로 답하는 법률 QA 어시스턴트다. "
            "질문자는 법률을 잘 모르는 일반인이라고 가정하고, 어려운 법률 용어는 쉬운 말로 풀어 설명한다. "
            "조문에 없는 내용은 추측하거나 지어내지 않는다. "
            "답변에는 근거가 되는 조·항·호·목을 명시하고, cited_articles 에는 컨텍스트에 실제로 있는 조문만 적는다. "
            "제공된 조문만으로 답할 수 없으면 is_answerable 을 false 로 하고 그 이유를 답변에 쓴다.",
        ),
        ("human", "조문:\n{context}\n\n질문: {question}"),
    ]
)


@dataclass
class RagResult:
    question: str
    answer: str
    is_answerable: bool
    citations: list[str]  # 제공된 컨텍스트에 실제 존재하는 조문만
    unverified_citations: list[str] = field(default_factory=list)  # 컨텍스트에 없는 조문(환각 의심)
    contexts: list[ParentContext] = field(default_factory=list)
    refused: bool = False  # 법령에 근거가 없어 답변을 거부했는가
    refusal_reason: str | None = None  # low_relevance | not_answerable | no_verified_citation


class RagPipeline:
    def __init__(
        self,
        retriever=None,
        reranker=None,
        llm=None,
        use_reranker: bool = True,
        min_relevance: int | None = RAG_MIN_RELEVANCE,
    ):
        self.min_relevance = min_relevance
        self.retriever = retriever or HybridRetriever()
        self.reranker = (reranker or LLMReranker()) if use_reranker else None
        llm = llm or get_llm_model(temperature=0, max_tokens=1024)
        # LLM 라우터는 function_calling 방식에서만 구조화 출력이 동작한다 (notebook/qdrant_test.ipynb).
        self.chain = _PROMPT | llm.with_structured_output(_RagAnswer, method="function_calling")

    def ask(
        self, question: str, retrieve_k: int = 20, rerank_n: int = 6, max_parents: int = 4
    ) -> RagResult:
        hits = self.retriever.retrieve(question, top_k=retrieve_k)
        if self.reranker:
            hits = self.reranker.rerank(question, hits, top_n=rerank_n)
        contexts = group_by_parent(hits, max_parents=max_parents)
        if not contexts:
            return self._refuse(question, "low_relevance", contexts)

        # 1차 거부: rerank 가 "질문과 관련된 조문이 없다"고 판단하면 답변 LLM 을 부르지 않는다.
        best = hits[0].rerank_score if self.reranker and hits else None
        if self.min_relevance is not None and best is not None and best < self.min_relevance:
            return self._refuse(question, "low_relevance", contexts)

        context_text = "\n\n".join(f"[{c.citation}]\n{c.text}" for c in contexts)
        out: _RagAnswer = self.chain.invoke({"context": context_text, "question": question})

        # 2차 거부: LLM 이 근거 없음으로 판단했거나, 인용한 조문이 컨텍스트에 실제로 없으면 답을 내지 않는다.
        provided = {c.article for c in contexts}
        verified, unverified = [], []
        for cite in out.cited_articles:
            (verified if cited_article(cite) in provided else unverified).append(cite)
        if not out.is_answerable:
            return self._refuse(question, "not_answerable", contexts)
        if not verified:
            return self._refuse(question, "no_verified_citation", contexts)
        return RagResult(question, out.answer, True, verified, unverified, contexts)

    @staticmethod
    def _refuse(question: str, reason: str, contexts: list[ParentContext]) -> RagResult:
        return RagResult(
            question, REFUSAL_MESSAGE, False, [], contexts=contexts, refused=True, refusal_reason=reason
        )


# ----- CLI ----------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag.pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_parse = sub.add_parser("parse", help="법령 XML → Canonical JSON 만 생성")
    p_parse.add_argument("--refresh", action="store_true", help="캐시를 무시하고 API 에서 다시 받기")
    p_ing = sub.add_parser("ingest", help="파싱 → 청킹 → 임베딩 → Qdrant 적재")
    p_ing.add_argument("--refresh", action="store_true", help="캐시를 무시하고 API 에서 다시 받기")
    p_ing.add_argument("--recreate", action="store_true", help=f"컬렉션 '{COLLECTION_NAME}' 을 지우고 다시 생성")
    p_ask = sub.add_parser("ask", help="질문하기")
    p_ask.add_argument("question")
    p_ask.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    if args.cmd == "parse":
        doc = build_canonical(args.refresh)
        print(f"조문 {len(doc['articles'])}개 → {CANONICAL_JSON_PATH}")
    elif args.cmd == "ingest":
        print(ingest(refresh=args.refresh, recreate=args.recreate))
    else:
        r = RagPipeline(use_reranker=not args.no_rerank).ask(args.question)
        print(f"\n{r.answer}\n")
        print(f"answerable: {r.is_answerable}" + (f" (거부: {r.refusal_reason})" if r.refused else ""))
        print(f"근거: {', '.join(r.citations) or '-'}")
        if r.unverified_citations:
            print(f"컨텍스트에 없는 인용(검증 실패): {', '.join(r.unverified_citations)}")
        print(f"검색된 조문: {', '.join(c.citation for c in r.contexts)}")


if __name__ == "__main__":
    _main()
