"""골든셋 평가.

    uv run python -m eval.evaluate                  # 운영 컬렉션을 현재 설정 그대로 평가
    uv run python -m eval.evaluate --rerank         # LLM rerank 까지 거친 최종 컨텍스트로 평가 + rerank 점수 분포(거부 기준 점수 보정용)
    uv run python -m eval.evaluate --report         # 평가 기준표(데이터 처리·후보 검색·최종 순위·최종 컨텍스트·단일 근거·거절)로 4개 구성 비교
    uv run python -m eval.evaluate --answers        # 답변 평가: 인용 정확성·정답성·근거 충실성·거절 정밀도/재현율 (LLM 호출)
    uv run python -m eval.evaluate --refusal        # 전체 파이프라인으로 답변 거부 평가 (범위 밖 질문 + 답해야 할 질문 표본)
    uv run python -m eval.evaluate --ablate         # 검색 모드 비교 (dense / sparse / hybrid RRF / hybrid DBSF)
    uv run python -m eval.evaluate --sweep          # 청킹·검색 변형 비교 (임시 컬렉션 eval_* 를 만들고 지운다)
    uv run python -m eval.evaluate --sweep --only base,cap2 --rerank

골든셋(eval/golden_set.jsonl)의 gold 는 정답 근거 [{"article": "제7조", "paragraph": 3}, ...] 이고,
paragraph 가 없으면 그 조 어디든 맞는 것으로 본다. gold 가 비어 있는 문항(oos)은 검색 지표에서 제외한다.

지표 (모두 질문 평균):
  cand@K    상위 K child 안에 정답 조가 하나라도 있는 비율         (rerank 가 볼 후보에 정답이 있는가)
  cand_all  정답 조 중 후보 안에 있는 비율                           (정답이 여럿인 문항의 회수율)
  art@3     중복을 뺀 상위 3개 조 안에 정답 조가 있는 비율          (rerank 전 순위 품질)
  mrr       정답 조가 처음 나오는 조 순위의 역수
  child@10  정답이 항까지 지정된 문항에서, 상위 10 child 안에 그 항의 child 가 있는 비율
  div       상위 K child 가 담고 있는 서로 다른 조의 평균 개수      (낮을수록 같은 조가 자리를 독점)
  ctx       (--rerank) 최종 컨텍스트(조 4개) 안에 정답 조가 하나라도 있는 비율
  ctx_all   (--rerank) 정답 조 중 최종 컨텍스트에 있는 비율
"""

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import RateLimitError
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from common.config import (
    API_KEY,
    BASE_DIR,
    BASE_URL,
    CANONICAL_JSON_PATH,
    EMBEDDING_MODEL,
    MODEL,
    RAG_MIN_RELEVANCE,
)
from common.qdrant import get_qdrant_client
from eval.metrics import (
    citation_scores,
    context_coverage,
    hit_at_k,
    ndcg_articles,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    refusal_precision_recall,
)
from rag.chunker import CIRCLED_TO_INT, chunk_law
from rag.loader import RE_NOTE, load_canonical
from rag.pipeline import RagPipeline, cited_article
from rag.reranker import LLMReranker
from rag.retriever import MODES, HybridRetriever, group_by_parent
from rag.vectorstore import index_chunks

GOLDEN_PATH = BASE_DIR / "data" / "golden_data_set" / "golden_set.jsonl"  # 직접 만든 골든셋 (answer_articles 는 조 단위 정답)
CACHE_PATH = BASE_DIR / "data" / "processed" / "eval_embed_cache.json"  # 임베딩 캐시(생성물)
RERANK_CACHE_PATH = BASE_DIR / "data" / "processed" / "eval_rerank_cache.json"  # rerank 점수 캐시(생성물)
COMPARE_OUT_PATH = BASE_DIR / "data" / "processed" / "eval_compare.json"  # 비교 결과 원자료
ANSWER_CACHE_PATH = BASE_DIR / "data" / "processed" / "eval_answers_cache.json"  # 답변·채점 캐시(생성물)
TEMP_PREFIX = "eval_"  # 이 접두어의 컬렉션만 평가 스크립트가 만들고 지운다
INT_TO_CIRCLED = {v: k for k, v in CIRCLED_TO_INT.items()}


@dataclass
class Variant:
    chunk: dict = field(default_factory=dict)  # chunk_law 인자: max_chars, resolve_refs
    cap: int | None = None  # 조당 child 상한 (retrieve 의 max_per_parent)


VARIANTS: dict[str, Variant] = {
    "base": Variant(),
    "cap2": Variant(cap=2),
    "cap3": Variant(cap=3),
    "refs": Variant({"resolve_refs": True}),
    "refs+cap2": Variant({"resolve_refs": True}, cap=2),
    "refs+cap3": Variant({"resolve_refs": True}, cap=3),
    "c300": Variant({"max_chars": 300}),
    "c800": Variant({"max_chars": 800}),
    "c1200": Variant({"max_chars": 1200}),
}


class CachedEmbeddings:
    """임베딩 호출을 캐시하고 호출 속도를 조절한다.

    - 같은 텍스트는 다시 임베딩하지 않는다. 캐시는 디스크에 저장해 중간에 실패해도 이어 받는다.
    - 라우터의 분당 호출 제한(30회)을 넘지 않도록 호출 간격을 넓게 두고, 429 면 기다렸다가 재시도한다.
    """

    MIN_INTERVAL = 4.0  # 초. 분당 약 15회 (제한의 절반: 라우터의 집계 방식이 불확실해 여유를 둔다)
    RETRY_WAIT = 70  # 초
    MAX_ATTEMPTS = 6

    def __init__(self, inner, cache_path: Path = CACHE_PATH):
        self.inner, self.cache_path, self._last = inner, cache_path, 0.0
        self.cache: dict[str, list[float]] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

    def _call(self, fn, arg):
        for attempt in range(self.MAX_ATTEMPTS):
            time.sleep(max(0.0, self._last + self.MIN_INTERVAL - time.monotonic()))
            try:
                return fn(arg)
            except RateLimitError:
                if attempt == self.MAX_ATTEMPTS - 1:
                    raise
                print(f"  (429 rate limit: {self.RETRY_WAIT}초 대기 후 재시도)", flush=True)
                time.sleep(self.RETRY_WAIT)
            finally:
                self._last = time.monotonic()

    def embed_documents(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            self.cache.update(zip(todo, self._call(self.inner.embed_documents, todo)))
            self._save()
        return [self.cache[t] for t in texts]

    def embed_query(self, text):
        if text not in self.cache:
            self.cache[text] = self._call(self.inner.embed_query, text)
            self._save()
        return self.cache[text]


def make_llm() -> ChatOpenAI:
    """평가용 LLM. 분당 호출 제한(30회)을 넘지 않도록 초당 0.2회(분당 12회)로 제한한다."""
    return ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
        use_responses_api=False,  # base url 사용 시 필요 (common/ai_model.py 와 동일)
        max_tokens=1024,
        # SDK 자동 재시도(기본 2회)는 429 때 로그 없이 요청을 늘려 제한 창을 계속 채운다. 재시도는 with_retry 한 곳에서만 한다.
        max_retries=0,
        rate_limiter=InMemoryRateLimiter(requests_per_second=0.2, max_bucket_size=1),
    )


def with_retry(fn, *args, wait=70, attempts=4):
    """429 면 기다렸다가 같은 호출을 다시 한다. 다른 예외는 그대로 낸다 (실패를 숨기지 않는다)."""
    for attempt in range(attempts):
        try:
            return fn(*args)
        except RateLimitError:
            if attempt == attempts - 1:
                raise
            print(f"  (LLM 429: {wait}초 대기 후 재시도)", flush=True)
            time.sleep(wait)


class NoRerank:
    """rerank 를 거치지 않는 구성: 1단계 검색 순서 그대로 상위 n개."""

    def rerank(self, query, hits, top_n=5):
        return hits[:top_n]


class CachedReranker:
    """LLM rerank 점수를 (검색 모드, 질문, 후보 구성) 단위로 디스크에 캐시한다. 같은 입력이면 LLM 을 다시 부르지 않는다."""

    def __init__(self, inner, mode: str, path: Path = RERANK_CACHE_PATH):
        self.inner, self.mode, self.path = inner, mode, path
        self.cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.llm_calls = 0

    def rerank(self, query, hits, top_n=5):
        key = f"{self.mode}|{query}|{','.join(h.chunk_id for h in hits)}"
        if key not in self.cache:
            with_retry(self.inner.rerank, query, hits, len(hits))  # 후보 전체에 점수가 기록된다
            self.cache[key] = {h.chunk_id: h.rerank_score for h in hits}
            self.llm_calls += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        scores = self.cache[key]
        for h in hits:
            h.rerank_score = scores.get(h.chunk_id)
        order = sorted(range(len(hits)), key=lambda i: (-(hits[i].rerank_score if hits[i].rerank_score is not None else -1), i))
        return [hits[i] for i in order[:top_n]]


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    """골든셋을 읽어 평가용 형태로 정규화한다.

    - 새 형식: answerable, answer_articles(조 단위 정답), answer_points(정답 요지), related_articles
    - 옛 형식: gold = [{"article": ..., "paragraph": ...}] 가 이미 있으면 그대로 쓴다.
    정답 조가 법령에 없으면 즉시 오류를 낸다 (라벨 오류가 점수를 왜곡하지 않게).
    """
    doc = load_canonical(CANONICAL_JSON_PATH)
    labels = {a["label"] for a in doc["articles"]} | {"부칙"}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "gold" not in r:
            answerable = r.get("answerable", True)
            r["gold"] = [{"article": a} for a in r.get("answer_articles", [])] if answerable else []
            if answerable and not r["gold"]:
                raise ValueError(f"{r['id']}: answerable 인데 answer_articles 가 비어 있다")
        bad = [g["article"] for g in r["gold"] if g["article"] not in labels]
        if bad:
            raise ValueError(f"{r['id']}: 법령에 없는 정답 조 {bad}")
        rows.append(r)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("골든셋 id 가 중복됐다")
    return rows


def _matches(hit_payload: dict, gold: dict) -> bool:
    if hit_payload["article"] != gold["article"]:
        return False
    return "paragraph" not in gold or hit_payload.get("paragraph") == INT_TO_CIRCLED[gold["paragraph"]]


def evaluate(
    retriever, golden, cap=None, top_k=20, reranker=None, rerank_n=6, max_parents=4, mode="hybrid"
) -> dict:
    rows, oos = [], []
    for q in golden:
        gold = q["gold"]
        if not gold:
            if reranker:  # 범위 밖 질문은 검색 지표에서 빼되, 거부 기준 점수 보정을 위해 rerank 점수만 모은다
                hits = retriever.retrieve(q["question"], top_k=top_k, max_per_parent=cap, mode=mode)
                top = with_retry(reranker.rerank, q["question"], hits, rerank_n)
                oos.append({"id": q["id"], "top_score": top[0].rerank_score if top else None})
            continue
        hits = retriever.retrieve(q["question"], top_k=top_k, max_per_parent=cap, mode=mode)
        gold_arts = {g["article"] for g in gold}
        order = list(dict.fromkeys(h.payload["article"] for h in hits))  # 중복 뺀 조 순서
        covered = gold_arts & set(order)
        first = next((i for i, a in enumerate(order, 1) if a in gold_arts), None)
        para_gold = [g for g in gold if "paragraph" in g]
        row = {
            "id": q["id"],
            "cand": bool(covered),
            "cand_all": len(covered) / len(gold_arts),
            "art3": bool(gold_arts & set(order[:3])),
            "mrr": 1 / first if first else 0.0,
            "div": len(order),
            "child10": (
                any(_matches(h.payload, g) for g in para_gold for h in hits[:10]) if para_gold else None
            ),
        }
        if reranker:
            top = with_retry(reranker.rerank, q["question"], hits, rerank_n)
            row["top_score"] = top[0].rerank_score if top else None
            ctx = {c.article for c in group_by_parent(top, max_parents)}
            row["ctx"] = bool(gold_arts & ctx)
            row["ctx_all"] = len(gold_arts & ctx) / len(gold_arts)
        rows.append(row)

    mean = lambda key: statistics.mean(float(r[key]) for r in rows)
    child = [r["child10"] for r in rows if r["child10"] is not None]
    summary = {
        f"cand@{top_k}": mean("cand"),
        "cand_all": mean("cand_all"),
        "art@3": mean("art3"),
        "mrr": mean("mrr"),
        "child@10": statistics.mean(map(float, child)) if child else float("nan"),
        "div": mean("div"),
    }
    if reranker:
        summary |= {"ctx": mean("ctx"), "ctx_all": mean("ctx_all")}
    return {"summary": summary, "rows": rows, "oos": oos}


def _print_table(results: dict[str, dict]) -> None:
    keys = list(next(iter(results.values()))["summary"])
    print(f"{'variant':<12}" + "".join(f"{k:>10}" for k in keys))
    for name, res in results.items():
        print(f"{name:<12}" + "".join(f"{res['summary'][k]:>10.3f}" for k in keys))


def _misses(res: dict, key: str) -> list[str]:
    return [r["id"] for r in res["rows"] if not r[key]]


def _print_score_calibration(res: dict) -> None:
    """rerank 최고 점수의 분포와, 기준 점수별 거부율/오거부율을 보여 준다."""
    ans = sorted(r["top_score"] for r in res["rows"] if r.get("top_score") is not None)
    oos = sorted(r["top_score"] for r in res["oos"] if r["top_score"] is not None)
    print(f"\n[rerank 최고 점수] 답해야 할 질문 {len(ans)}개: {ans}")
    print(f"[rerank 최고 점수] 범위 밖 질문 {len(oos)}개: {oos}")
    print(f"{'기준(미만 거부)':<14}{'범위밖 거부율':>12}{'오거부율(답해야 할 질문)':>26}")
    for t in range(1, 10):
        print(f"{t:<14}{sum(x < t for x in oos) / max(len(oos), 1):>12.2f}{sum(x < t for x in ans) / max(len(ans), 1):>26.2f}")


def run_refusal(client, embeddings, golden, min_relevance, n_answerable) -> None:
    """전체 파이프라인(검색 → rerank → 답변)으로 거부가 제대로 되는지 본다."""
    llm = make_llm()
    pipe = RagPipeline(
        retriever=HybridRetriever(client=client, embeddings=embeddings),
        reranker=LLMReranker(llm, strict=True),
        llm=llm,
        min_relevance=min_relevance,
    )
    oos = [q for q in golden if not q["gold"]]
    answerable = [q for q in golden if q["gold"]]
    step = max(1, len(answerable) // n_answerable)
    sample = answerable[::step][:n_answerable]  # 유형이 고루 섞이도록 간격을 두고 뽑는다
    print(f"기준 점수: {min_relevance} (미만이면 거부) | 범위 밖 {len(oos)}개 + 답해야 할 질문 표본 {len(sample)}개\n")

    refused_oos = refused_ans = cited_ok = 0
    for q in oos + sample:
        r = with_retry(pipe.ask, q["question"])
        is_oos = not q["gold"]
        mark = "거부" if r.refused else "답변"
        ok = r.refused if is_oos else not r.refused
        if is_oos:
            refused_oos += r.refused
        else:
            refused_ans += r.refused
            gold_arts = {g["article"] for g in q["gold"]}
            cited_ok += bool(gold_arts & {cited_article(c) for c in r.citations})
        print(f"{'OK ' if ok else 'NG '}{q['id']} [{q['type']}] {mark}({r.refusal_reason or '-'}) {q['question'][:30]}")
    print(f"\n범위 밖 질문 거부율: {refused_oos}/{len(oos)}")
    print(f"답해야 할 질문 오거부율: {refused_ans}/{len(sample)}")
    print(f"답한 질문 중 정답 조를 인용한 비율: {cited_ok}/{len(sample) - refused_ans}")


REPORT_CONFIGS = [  # (이름, 1단계 검색 모드, LLM rerank 사용 여부)
    ("dense", "dense", False),
    ("dense+rerank", "dense", True),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", RE_NOTE.sub("", text))


def data_processing_report(client) -> dict:
    """데이터 처리 단계: 원문 보존율과 ID 충돌 수. API 호출 없음."""
    import xml.etree.ElementTree as ET

    from common.config import COLLECTION_NAME, LAW_RAW_XML_PATH
    from rag.chunker import render_article

    doc = load_canonical(CANONICAL_JSON_PATH)
    chunks = chunk_law(doc)

    # 1) 원본 XML → 정규화 JSON: XML 의 모든 조문·항·호·목 본문이 JSON 에 있는가 (번호 접두어·개정 표기는 제외하고 비교)
    labels = {
        "조문내용": re.compile(r"^제\d+조(?:의\d+)?(?:\([^)]*\))?"),
        "항내용": re.compile(r"^[①-㊿]"),
        "호내용": re.compile(r"^\d+(?:의\d+)*\."),
        "목내용": re.compile(r"^[가-힣](?:의\d+)?\."),
    }
    root = ET.parse(LAW_RAW_XML_PATH).getroot()
    xml_leaves = []
    for unit in root.find("조문").findall("조문단위"):
        if (unit.findtext("조문여부") or "").strip() != "조문":
            continue
        for tag, rx in labels.items():
            for el in unit.iter(tag):
                t = _norm(rx.sub("", (el.text or "").strip(), count=1))
                if t:
                    xml_leaves.append(t)

    leaves = []

    def walk(node):
        if node.get("text"):
            leaves.append(_norm(node["text"]))
        for key in ("items", "subitems", "paragraphs"):
            for child in node.get(key, []):
                walk(child)

    for art in doc["articles"]:
        walk(art)
    canon_blob = "".join(leaves)
    chunk_blob = _norm("".join(c.text for c in chunks))
    parent_blob = _norm("".join(c.parent_text for c in chunks))

    def kept(items, blob):
        return sum(len(x) for x in items if x in blob) / sum(len(x) for x in items)

    chunk_ids = [c.chunk_id for c in chunks]
    point_ids = [c.point_id for c in chunks]
    stored = client.get_collection(COLLECTION_NAME).points_count
    return {
        "xml_to_canonical": kept(xml_leaves, canon_blob),
        "canonical_to_chunk": kept(leaves, chunk_blob),
        "canonical_to_parent": kept(leaves, parent_blob),
        "xml_leaves": len(xml_leaves),
        "canonical_leaves": len(leaves),
        "chunks": len(chunks),
        "chunk_id_collisions": len(chunk_ids) - len(set(chunk_ids)),
        "point_id_collisions": len(point_ids) - len(set(point_ids)),
        "qdrant_points": stored,
        "qdrant_missing": len(chunks) - stored,
    }


def _ctx_stats(contexts, gold) -> dict:
    cov = context_coverage({c.article for c in contexts}, gold)
    return {"coverage": cov, "full": cov == 1.0, "parents": len(contexts)}


def run_report(
    client, embeddings, golden, threshold: int | None, config_names: list[str] | None = None, skip_oos: bool = False
) -> None:
    """평가 기준표(데이터 처리 / 후보 검색 / 최종 순위 / 최종 컨텍스트 / 단일 근거 / 거절)에 따라 4개 구성을 비교한다."""
    corpus = [c.payload() for c in chunk_law(load_canonical(CANONICAL_JSON_PATH))]
    llm_reranker = LLMReranker(make_llm(), strict=True)
    mean = lambda xs: statistics.mean(xs) if xs else float("nan")

    # 정답이 조 단위뿐이면 조 단위로 합친 순위에서, 항까지 지정돼 있으면 child 단위(등급 2/1/0)로 nDCG 를 계산한다.
    article_level = not any("paragraph" in g for q in golden for g in q["gold"])
    ndcg = (lambda ranked, gold, k: ndcg_articles(ranked, gold, k)) if article_level else (lambda ranked, gold, k: ndcg_at_k(ranked, gold, corpus, k))

    dp = data_processing_report(client)
    print("[데이터 처리]")
    print(f"  원문 보존율 XML→정규화JSON {dp['xml_to_canonical']:.4f} ({dp['xml_leaves']}개 단위) | 정규화JSON→child {dp['canonical_to_chunk']:.4f} | →parent {dp['canonical_to_parent']:.4f}")
    print(f"  ID 충돌: chunk_id {dp['chunk_id_collisions']}, point_id {dp['point_id_collisions']} | Qdrant 저장 {dp['qdrant_points']}/{dp['chunks']} (누락 {dp['qdrant_missing']})")

    out = {"data_processing": dp, "configs": {}}
    selected = [c for c in REPORT_CONFIGS if config_names is None or c[0] in config_names]
    for name, mode, use_rr in selected:
        retriever = HybridRetriever(client=client, embeddings=embeddings)
        reranker = CachedReranker(llm_reranker, mode) if use_rr else NoRerank()
        rows, oos_scores, oos_detail = [], [], []
        for q in golden:
            if not q["gold"] and (skip_oos or not use_rr):
                continue  # 범위 밖 질문은 rerank 점수(거절 문지기 보정)에만 필요하다. skip_oos 면 LLM 을 부르지 않는다.
            hits30 = retriever.retrieve(q["question"], top_k=30, mode=mode)
            hits20 = hits30[:20]  # 운영과 같이 후보 20개를 rerank 에 넘긴다
            final20 = reranker.rerank(q["question"], hits20, 20)  # rerank 순서로 정렬된 후보 전체 (rerank 점수는 캐시)
            final10 = final20[:10]
            if not q["gold"]:
                if use_rr:
                    oos_scores.append(final10[0].rerank_score)
                    oos_detail.append({"id": q["id"], "score": final10[0].rerank_score, "top_article": final10[0].payload["article"]})
                continue
            gold = q["gold"]
            cand = [h.payload for h in hits30]
            ranked = [h.payload for h in final10]
            ctx_arts = {c.article for c in group_by_parent(final10[:6], 4)}
            cov = context_coverage(ctx_arts, gold)
            row = {
                "id": q["id"], "type": q["type"], "n_gold": len(gold),
                "recall20": recall_at_k(cand, gold, 20), "recall30": recall_at_k(cand, gold, 30),
                "recall20_art": recall_at_k(cand, gold, 20, strict=False), "recall30_art": recall_at_k(cand, gold, 30, strict=False),
                "ndcg5": ndcg(ranked, gold, 5), "ndcg10": ndcg(ranked, gold, 10),
                "ndcg5_child": ndcg_at_k(ranked, gold, corpus, 5), "ndcg10_child": ndcg_at_k(ranked, gold, corpus, 10),
                "ctx_cov": cov, "ctx_full": cov == 1.0,
                "rr10": reciprocal_rank(ranked, gold, 10),
                "hit": {k: hit_at_k(ranked, gold, k) for k in (1, 3, 5, 10)},
                "score": final10[0].rerank_score if use_rr else None,
                # LLM 에 넘기는 조를 고르는 방식별 근거 확보: "current"=상위 child 6개에서 조 최대 4개, 3/4/5=후보 전체에서 상위 조 N개
                "ctx_select": {
                    "current": _ctx_stats(group_by_parent(final20[:6], 4), gold),
                    **{str(k): _ctx_stats(group_by_parent(final20, k), gold) for k in (3, 4, 5)},
                },
            }
            rows.append(row)
        out["configs"][name] = {"rows": rows, "oos_scores": oos_scores, "oos": oos_detail, "llm_calls": getattr(reranker, "llm_calls", 0)}

    # 컨텍스트 크기 민감도(hybrid, rerank 없이): LLM 에 넘기는 조를 늘리면 확보율이 얼마나 오르는가. API 호출 없음.
    if any(c[0] == "hybrid" for c in selected):
        retriever = HybridRetriever(client=client, embeddings=embeddings)
        answerable = [q for q in golden if q["gold"]]
        cand = {q["id"]: retriever.retrieve(q["question"], top_k=20, mode="hybrid") for q in answerable}
        sweep = []
        for n, p in [(6, 4), (8, 4), (20, 3), (20, 4), (20, 5), (20, 8)]:  # child 20 = 후보 전체에서 상위 조 P개만 고르는 방식
            cov, sizes = [], []
            for q in answerable:
                ctx = group_by_parent(cand[q["id"]][:n], p)
                cov.append(context_coverage({c.article for c in ctx}, q["gold"]))
                sizes.append(len(ctx))
            multi = [c == 1.0 for c, q in zip(cov, answerable) if len(q["gold"]) >= 2]
            sweep.append({"child_n": n, "max_parents": p, "coverage": mean(cov), "full": mean([float(c == 1.0) for c in cov]),
                          "full_multi": mean([float(x) for x in multi]), "avg_parents": mean(sizes)})
        out["ctx_size_sweep"] = sweep

    COMPARE_OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    cfgs = out["configs"]
    names = list(cfgs)
    single = lambda rows: [r for r in rows if r["n_gold"] == 1]
    multi = lambda rows: [r for r in rows if r["n_gold"] >= 2]
    n_all = len(cfgs[names[0]]["rows"])
    print(f"\n정답 있는 질문 {n_all}개 (근거 1개 {len(single(cfgs[names[0]]['rows']))}개, 근거 2개 이상 {len(multi(cfgs[names[0]]['rows']))}개)")

    def table(title, heads, getters):
        print(f"\n[{title}]")
        print(f"{'구성':<15}" + "".join(f"{h:>13}" for h in heads))
        for name in names:
            rows = cfgs[name]["rows"]
            print(f"{name:<15}" + "".join(f"{g(rows):>13.3f}" for g in getters))

    if any("paragraph" in g for q in golden for g in q["gold"]):
        table("후보 검색: 필수 근거 Recall (항 단위 | 조 단위)", ["R@20(항)", "R@30(항)", "R@20(조)", "R@30(조)"],
              [lambda r, k=k: mean([x[k] for x in r]) for k in ("recall20", "recall30", "recall20_art", "recall30_art")])
    else:  # 정답이 조 단위뿐이면 항 단위 값은 조 단위와 같다
        table("후보 검색: 필수 근거 Recall (조 단위)", ["Recall@20", "Recall@30"],
              [lambda r, k=k: mean([x[k] for x in r]) for k in ("recall20", "recall30")])
    table("최종 순위: nDCG (조 단위로 합친 순위, 정답 조=1)" if article_level else "최종 순위: nDCG (정답 근거=2, 같은 조의 다른 항=1)", ["nDCG@5", "nDCG@10"],
          [lambda r, k=k: mean([x[k] for x in r]) for k in ("ndcg5", "ndcg10")])
    table("최종 컨텍스트: 필수 근거 확보 (조 4개)", ["확보율", "전부확보(전체)", "전부확보(다중)"],
          [lambda r: mean([x["ctx_cov"] for x in r]), lambda r: mean([float(x["ctx_full"]) for x in r]),
           lambda r: mean([float(x["ctx_full"]) for x in multi(r)])])
    table("단일 근거 질문: 첫 정답 순위", ["MRR@10", "Hit@1", "Hit@3", "Hit@5", "Hit@10"],
          [lambda r: mean([x["rr10"] for x in single(r)])] + [lambda r, k=k: mean([float(x["hit"][k]) for x in single(r)]) for k in (1, 3, 5, 10)])

    print("\n[컨텍스트를 고르는 방식] 현재(상위 child 6개 → 조 ≤4) vs 후보 전체에서 상위 조 N개")
    print(f"{'구성':<15}{'방식':<22}{'확보율':>8}{'전부확보':>9}{'전부(다중)':>11}{'평균 조 수':>10}")
    for name in names:
        rows = cfgs[name]["rows"]
        for key, label in (("current", "child 6 → 조 ≤4 (현재)"), ("3", "상위 조 3개"), ("4", "상위 조 4개"), ("5", "상위 조 5개")):
            sel = [r["ctx_select"][key] for r in rows]
            multi = [x["full"] for x, r in zip(sel, rows) if r["n_gold"] >= 2]
            print(f"{name:<15}{label:<22}{mean([x['coverage'] for x in sel]):>8.3f}{mean([float(x['full']) for x in sel]):>9.3f}"
                  f"{mean([float(m) for m in multi]):>11.3f}{mean([x['parents'] for x in sel]):>10.2f}")

    if "ctx_size_sweep" in out:
        print("\n[컨텍스트 크기 민감도: hybrid, rerank 없이] 상위 child N개 → 조 최대 P개")
        print(f"{'child N':>8}{'조 최대':>8}{'확보율':>9}{'전부확보(전체)':>14}{'전부확보(다중)':>14}{'평균 조 수':>10}")
        for r in out["ctx_size_sweep"]:
            mark = "  ← 현재" if (r["child_n"], r["max_parents"]) == (6, 4) else ""
            print(f"{r['child_n']:>8}{r['max_parents']:>8}{r['coverage']:>9.3f}{r['full']:>14.3f}{r['full_multi']:>14.3f}{r['avg_parents']:>10.2f}{mark}")

    final_name = "hybrid+rerank" if "hybrid+rerank" in cfgs else names[-1]
    print(f"\n[유형별: {final_name}]")
    print(f"{'유형':<10}{'문항':>4}{'Recall@20':>11}{'nDCG@5':>9}{'Hit@1':>8}{'확보율':>8}{'전부확보':>9}")
    by = {}
    for r in cfgs[final_name]["rows"]:
        by.setdefault(r["type"], []).append(r)
    for t, rows in by.items():
        print(f"{t:<10}{len(rows):>4}{mean([x['recall20'] for x in rows]):>11.3f}{mean([x['ndcg5'] for x in rows]):>9.3f}"
              f"{mean([float(x['hit'][1]) for x in rows]):>8.3f}{mean([x['ctx_cov'] for x in rows]):>8.3f}{mean([float(x['ctx_full']) for x in rows]):>9.3f}")

    print("\n[거절 문지기(1차, rerank 점수)] 정답 질문 vs 범위 밖 질문의 최고 점수")
    for name in [n for n in names if n.endswith("+rerank")]:
        ans = sorted(r["score"] for r in cfgs[name]["rows"] if r["score"] is not None)
        oos = sorted(x for x in cfgs[name]["oos_scores"] if x is not None)
        if not oos:
            print(f"{name:<15} 정답 질문 최고 점수: min={ans[0]}, 10점 {sum(v == 10 for v in ans)}/{len(ans)} | 범위 밖 질문을 평가하지 않아(--skip-oos) 거절 지표는 계산하지 않음")
            continue
        line = f"{name:<15} 정답: min={ans[0]}, 10점 {sum(v == 10 for v in ans)}/{len(ans)} | 범위밖: {oos}"
        if threshold is not None:
            rec = [(False, v < threshold) for v in ans] + [(True, v < threshold) for v in oos]
            pr, rc = refusal_precision_recall(rec)
            line += f" | 기준 {threshold}: 정밀도 {pr:.2f}, 재현율 {rc:.2f}"
        print(line)
    for a, b in (("dense", "dense+rerank"), ("hybrid", "hybrid+rerank")):
        if a in cfgs and b in cfgs:
            q_text = {x["id"]: x["question"] for x in golden}
            diff = [(x, y) for x, y in zip(cfgs[a]["rows"], cfgs[b]["rows"])
                    if (x["hit"][1], round(x["ctx_cov"], 3), round(x["ndcg5"], 3)) != (y["hit"][1], round(y["ctx_cov"], 3), round(y["ndcg5"], 3))]
            print(f"\n[{a} → {b}] 결과가 달라진 문항 {len(diff)}개 (Hit@1 / 컨텍스트 확보 / nDCG@5)")
            for x, y in diff:
                print(f"  {x['id']} [{x['type']}] Hit@1 {int(x['hit'][1])}→{int(y['hit'][1])} | 확보 {x['ctx_cov']:.2f}→{y['ctx_cov']:.2f} | nDCG@5 {x['ndcg5']:.2f}→{y['ndcg5']:.2f} | {q_text[x['id']][:30]}")

    print("\nLLM 호출(rerank, 새로 호출한 수):", {n: cfgs[n]["llm_calls"] for n in names})


class _Judgement(BaseModel):
    correctness: int = Field(ge=0, le=2, description="정답성. 0=정답 요지와 무관하거나 모순, 1=요지 일부만 포함, 2=정답 요지를 모두 정확히 포함")
    faithfulness: int = Field(ge=0, le=2, description="근거 충실성. 0=제공된 조문에 없는 내용을 지어냄, 1=대체로 근거가 있으나 일부 과장·누락, 2=모든 주장이 제공된 조문에 근거")
    reason: str = Field(description="판단 이유 한 문장")


_JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 법령 QA 답변을 채점하는 엄격한 평가자다. 두 가지를 채점한다.\n"
            "- 정답성: 답변이 [정답 요지]를 빠짐없이 정확하게 담았는가. 요지와 모순되면 0점.\n"
            "- 근거 충실성: 답변의 모든 주장이 [모델에 제공된 조문]에서 뒷받침되는가. 조문에 없는 내용을 덧붙였으면 감점한다.",
        ),
        (
            "human",
            "[질문]\n{question}\n\n[정답 요지]\n{points}\n\n[모델에 제공된 조문]\n{context}\n\n[답변]\n{answer}",
        ),
    ]
)


def run_answers(client, embeddings, golden, threshold, n_sample: int) -> None:
    """운영 구성(hybrid + rerank)으로 답변을 만들고 인용 정확성·정답성·근거 충실성·거절을 평가한다. LLM 호출이 발생한다."""
    llm = make_llm()
    cache = json.loads(ANSWER_CACHE_PATH.read_text(encoding="utf-8")) if ANSWER_CACHE_PATH.exists() else {}
    pipe = RagPipeline(
        retriever=HybridRetriever(client=client, embeddings=embeddings),
        reranker=CachedReranker(LLMReranker(llm, strict=True), "hybrid"),
        llm=llm,
        min_relevance=threshold,
    )
    judge = _JUDGE_PROMPT | llm.with_structured_output(_Judgement, method="function_calling")

    by_type: dict[str, list[dict]] = {}
    for q in golden:
        if q["gold"]:
            by_type.setdefault(q["type"], []).append(q)
    per_type = max(1, n_sample // max(len(by_type), 1))
    # 유형마다 같은 수를 앞에서부터 뽑는다(API 절약을 위한 표본). 전체를 보려면 --n-answerable 을 키운다.
    sample = [q for rows in by_type.values() for q in rows[:per_type]]
    oos = [q for q in golden if not q["gold"]]
    print(f"표본: 답해야 할 질문 {len(sample)}개 + 범위 밖 {len(oos)}개 | 기준 점수 {threshold}\n")

    def save():
        ANSWER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        ANSWER_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    records, cit_p, cit_r, correct, faith = [], [], [], [], []
    for q in sample + oos:
        key = f"{threshold}|{q['question']}"
        if key not in cache:
            r = with_retry(pipe.ask, q["question"])
            entry = {"answer": r.answer, "refused": r.refused, "reason": r.refusal_reason, "citations": r.citations,
                     "context": "\n\n".join(f"[{c.citation}]\n{c.text}" for c in r.contexts)}
            if q["gold"] and not r.refused:
                j = with_retry(
                    judge.invoke,
                    {"question": q["question"], "points": "\n".join(f"- {p}" for p in q.get("answer_points", [])), "context": entry["context"], "answer": r.answer},
                )
                entry["judge"] = j.model_dump()
            cache[key] = entry
            save()
        e = cache[key]
        is_oos = not q["gold"]
        records.append((is_oos, e["refused"]))
        tag = "거절" if e["refused"] else "답변"
        extra = ""
        if q["gold"] and not e["refused"]:
            p, rc = citation_scores(e["citations"], q["gold"])
            cit_p.append(p), cit_r.append(rc)
            j = e["judge"]
            correct.append(j["correctness"]), faith.append(j["faithfulness"])
            extra = f" 인용(P{p:.1f}/R{rc:.1f}) 정답성{j['correctness']} 충실성{j['faithfulness']}"
        ok = "OK " if (e["refused"] == is_oos) else "NG "
        print(f"{ok}{q['id']} [{q['type']}] {tag}({e['reason'] or '-'}){extra} {q['question'][:26]}")

    pr, rc = refusal_precision_recall(records)
    n = len(correct)
    print(f"\n[답변] 답한 질문 {n}개")
    print(f"  인용 정확성: 정밀도 {mean_(cit_p):.3f}, 재현율 {mean_(cit_r):.3f}")
    print(f"  정답성: 평균 {mean_(correct) / 2:.3f} (2점 만점 환산), 정답(2점) {sum(c == 2 for c in correct)}/{n}, 오답(0점) {sum(c == 0 for c in correct)}/{n}")
    print(f"  근거 충실성: 평균 {mean_(faith) / 2:.3f}, 충실(2점) {sum(f == 2 for f in faith)}/{n}, 환각(0점) {sum(f == 0 for f in faith)}/{n}")
    print(f"[거절] 정밀도 {pr:.3f} (거절한 것 중 실제 답변 불가능), 재현율 {rc:.3f} (답변 불가능 질문 중 거절)")


def mean_(xs):
    return statistics.mean(xs) if xs else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.evaluate")
    parser.add_argument("--sweep", action="store_true", help="변형 비교 (임시 컬렉션 사용)")
    parser.add_argument("--ablate", action="store_true", help="검색 모드 비교: dense / sparse / hybrid(RRF) / hybrid(DBSF)")
    parser.add_argument("--only", help="쉼표로 구분한 변형 이름만 실행")
    parser.add_argument("--rerank", action="store_true", help="LLM rerank 후 최종 컨텍스트 지표 포함")
    parser.add_argument("--cap", type=int, help="(비 sweep) 조당 child 상한")
    parser.add_argument("--keep", action="store_true", help="sweep 임시 컬렉션을 지우지 않는다")
    parser.add_argument("--refusal", action="store_true", help="전체 파이프라인으로 답변 거부 평가")
    parser.add_argument("--configs", default="dense,hybrid,hybrid+rerank", help="--report 에서 비교할 구성(쉼표). dense+rerank 는 LLM 호출이 더 든다")
    parser.add_argument("--skip-oos", action="store_true", help="--report 에서 범위 밖 질문의 rerank 호출을 생략(API 절약)")
    parser.add_argument("--answers", action="store_true", help="답변 평가(인용 정확성·정답성·근거 충실성·거절). LLM 호출 발생")
    parser.add_argument("--report", action="store_true", help="평가 기준표대로 dense / dense+rerank / hybrid / hybrid+rerank 비교")
    parser.add_argument("--min-relevance", type=int, default=RAG_MIN_RELEVANCE, help="거부 기준 rerank 점수(미만이면 거부)")
    parser.add_argument("--n-answerable", type=int, default=14, help="--answers/--refusal 에서 답해야 할 질문 표본 수(유형별로 균등 배분)")
    args = parser.parse_args()

    golden = load_golden()
    reranker = None
    if args.rerank:
        reranker = LLMReranker(make_llm(), strict=True)
    client = get_qdrant_client()
    # SDK 의 자체 재시도(기본 2회)도 호출 횟수에 잡혀 제한을 스스로 채우므로 끄고, 재시도는 래퍼 한 곳에서만 한다.
    embeddings = CachedEmbeddings(
        OpenAIEmbeddings(api_key=API_KEY, base_url=BASE_URL, model=EMBEDDING_MODEL, max_retries=0)
    )
    embeddings.embed_documents([q["question"] for q in golden])  # 질문 전체를 한 번의 호출로 임베딩

    if args.answers:
        run_answers(client, embeddings, golden, args.min_relevance, args.n_answerable)
        return
    if args.report:
        run_report(client, embeddings, golden, args.min_relevance, args.configs.split(",") if args.configs else None, args.skip_oos)
        return
    if args.refusal:
        run_refusal(client, embeddings, golden, args.min_relevance, args.n_answerable)
        return

    results: dict[str, dict] = {}
    if args.ablate:
        retriever = HybridRetriever(client=client, embeddings=embeddings)
        for mode in MODES:
            results[mode] = evaluate(retriever, golden, cap=args.cap, reranker=reranker, mode=mode)
    elif not args.sweep:
        retriever = HybridRetriever(client=client, embeddings=embeddings)
        results["current"] = evaluate(retriever, golden, cap=args.cap, reranker=reranker)
    else:
        doc = load_canonical(CANONICAL_JSON_PATH)
        names = args.only.split(",") if args.only else list(VARIANTS)
        created: list[str] = []
        try:
            for name in names:
                v = VARIANTS[name]
                col = f"{TEMP_PREFIX}{name.replace('+', '_')}"
                created.append(col)
                index_chunks(
                    chunk_law(doc, **v.chunk), client, embeddings, name=col, recreate=True, batch_size=64
                )
                retriever = HybridRetriever(client=client, embeddings=embeddings, name=col)
                results[name] = evaluate(retriever, golden, cap=v.cap, reranker=reranker)
        finally:
            if not args.keep:
                for col in created:
                    if col.startswith(TEMP_PREFIX) and client.collection_exists(col):
                        client.delete_collection(col)

    _print_table(results)
    if reranker and not args.sweep and not args.ablate:
        _print_score_calibration(results["current"])
    for name, res in results.items():
        miss = _misses(res, "cand")
        extra = f" | 최종 컨텍스트 실패: {_misses(res, 'ctx')}" if reranker else ""
        print(f"[{name}] 후보에 정답 조가 없는 문항: {miss}{extra}")


if __name__ == "__main__":
    main()
