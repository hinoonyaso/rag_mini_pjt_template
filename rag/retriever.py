"""Hybrid Retrieval: dense + sparse 후보를 Qdrant 에서 RRF 로 합치고, child 히트를 parent(조)로 묶는다."""

from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from common.ai_model import get_embedding_model
from common.config import COLLECTION_NAME
from common.qdrant import get_qdrant_client
from rag.vectorstore import DENSE, SPARSE, SparseEncoder


MODES = ("hybrid", "hybrid_dbsf", "dense", "sparse")
_TIE_MARGIN = 10  # 동점 정렬을 위해 top_k 보다 더 받아 온다


@dataclass
class Hit:
    chunk_id: str
    score: float
    payload: dict
    rerank_score: int | None = None  # LLM rerank 가 매긴 0~10 점수 (rerank 를 거치지 않았으면 None)

    @property
    def text(self) -> str:
        return self.payload["text"]

    @property
    def citation(self) -> str:
        return self.payload["citation"]


@dataclass
class ParentContext:
    """LLM 에 근거로 넘길 조(條) 단위 컨텍스트."""

    parent_id: str
    article: str  # 예: 제6조
    text: str  # 조 전체 원문
    score: float  # 포함된 child 중 가장 높은 순위의 점수
    matched: list[str] = field(default_factory=list)  # 실제로 검색된 child 의 인용 표기

    @property
    def citation(self) -> str:
        return self.parent_id.replace(":", " ")


class HybridRetriever:
    def __init__(
        self,
        client: QdrantClient | None = None,
        embeddings=None,
        name: str = COLLECTION_NAME,
        prefetch_k: int = 30,
    ):
        self.client = client or get_qdrant_client()
        self.embeddings = embeddings or get_embedding_model()
        self.name = name
        self.prefetch_k = prefetch_k
        self.sparse = SparseEncoder()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        query_filter: models.Filter | None = None,
        max_per_parent: int | None = None,
        mode: str = "hybrid",
    ) -> list[Hit]:
        """top_k 개의 child 를 돌려준다.

        mode: "hybrid"(dense+sparse, RRF 융합, 기본) | "hybrid_dbsf"(분포 기반 점수 융합) | "dense" | "sparse"
        max_per_parent 가 있으면 한 조(parent)에서 그 수까지만 담는다.
        """
        if mode not in MODES:
            raise ValueError(f"mode 는 {MODES} 중 하나여야 한다: {mode}")
        dense = self.embeddings.embed_query(query)
        fetch = (top_k if max_per_parent is None else min(top_k * 3, self.prefetch_k * 2)) + _TIE_MARGIN
        if mode in ("dense", "sparse"):
            result = self.client.query_points(
                collection_name=self.name,
                query=dense if mode == "dense" else self.sparse.encode(query),
                using=DENSE if mode == "dense" else SPARSE,
                query_filter=query_filter,
                limit=fetch,
                with_payload=True,
            )
        else:
            fusion = models.Fusion.RRF if mode == "hybrid" else models.Fusion.DBSF
            result = self.client.query_points(
                collection_name=self.name,
                prefetch=[
                    models.Prefetch(
                        query=dense, using=DENSE, limit=self.prefetch_k, filter=query_filter
                    ),
                    models.Prefetch(
                        query=self.sparse.encode(query),
                        using=SPARSE,
                        limit=self.prefetch_k,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=fusion),
                limit=fetch,
                with_payload=True,
            )
        hits = [Hit(p.payload["chunk_id"], p.score, p.payload) for p in result.points]
        # RRF 점수는 동점이 흔한데 Qdrant 는 동점 순서를 고정하지 않아 같은 질문도 결과가 달라진다.
        # 점수가 같으면 chunk_id 순으로 고정해, 같은 질문은 항상 같은 결과가 나오게 한다.
        hits.sort(key=lambda h: (-round(h.score, 9), h.chunk_id))
        if max_per_parent is None:
            return hits[:top_k]
        taken: dict[str, int] = {}
        capped = []
        for h in hits:
            pid = h.payload["parent_id"]
            if taken.get(pid, 0) < max_per_parent:
                taken[pid] = taken.get(pid, 0) + 1
                capped.append(h)
        return capped[:top_k]


def group_by_parent(hits: list[Hit], max_parents: int | None = None) -> list[ParentContext]:
    """child 히트를 parent(조)로 묶는다. 히트 순서(상위 순위 우선)를 유지하고 같은 조는 한 번만 담는다."""
    parents: dict[str, ParentContext] = {}
    for h in hits:
        pid = h.payload["parent_id"]
        ctx = parents.get(pid)
        if ctx is None:
            ctx = parents[pid] = ParentContext(
                parent_id=pid,
                article=h.payload["article"],
                text=h.payload["parent_text"],
                score=h.score,
            )
        ctx.matched.append(h.citation)
    result = list(parents.values())
    return result[:max_parents] if max_parents else result
