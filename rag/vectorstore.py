"""Embedding + Qdrant 컬렉션(dense + sparse) 생성 및 인덱싱.

컬렉션은 named vector 두 개를 가진다.
- "dense"  : 임베딩 모델 벡터 (cosine)
- "sparse" : 키워드 매칭용 sparse 벡터 (Qdrant IDF modifier 로 희귀 용어 가중)
"""

import math
import re
import zlib
from collections import Counter

from qdrant_client import QdrantClient, models

from common.ai_model import get_embedding_model
from common.config import COLLECTION_NAME
from common.qdrant import get_qdrant_client
from rag.chunker import Chunk

DENSE = "dense"
SPARSE = "sparse"

_RE_WORD = re.compile(r"[가-힣]+|[A-Za-z]+|\d+")
# "제6조", "제17조의2", "제2항", "제4호의2" 같은 조문 참조는 통째로 하나의 토큰으로 둔다.
_RE_REF = re.compile(r"제\d+(?:조|항|호|목)(?:의\d+)?")


class SparseEncoder:
    """형태소 분석기 없이 한국어 조사 변형에 견디는 sparse 인코더.

    한글 어절은 어절 전체 + 글자 bigram 으로 쪼개고(예: '인공지능을' → '인공','공지','지능','능을'),
    영문·숫자는 소문자 토큰, 조문 참조('제6조')는 하나의 토큰으로 만든다.
    토큰은 crc32 로 uint32 인덱스에 해시한다(프로세스와 무관하게 결정적).
    인덱싱과 질의에 같은 인코더를 써야 한다.
    """

    def features(self, text: str) -> Counter:
        feats: Counter = Counter(_RE_REF.findall(text))
        for word in _RE_WORD.findall(text):
            if word.isascii():
                feats[word.lower()] += 1
                continue
            if len(word) >= 2:
                feats[word] += 1
            feats.update(word[i : i + 2] for i in range(len(word) - 1))
        return feats

    def encode(self, text: str) -> models.SparseVector:
        merged: dict[int, float] = {}
        for token, tf in self.features(text).items():
            idx = zlib.crc32(token.encode("utf-8"))
            merged[idx] = merged.get(idx, 0.0) + 1.0 + math.log(tf)
        indices = sorted(merged)
        return models.SparseVector(indices=indices, values=[merged[i] for i in indices])


def embed_texts(texts: list[str], embeddings=None, batch_size: int = 32) -> list[list[float]]:
    embeddings = embeddings or get_embedding_model()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        vectors.extend(embeddings.embed_documents(texts[i : i + batch_size]))
    return vectors


def ensure_collection(
    client: QdrantClient, dense_size: int, name: str = COLLECTION_NAME, recreate: bool = False
) -> None:
    """컬렉션이 없으면 만든다. recreate=True 면 같은 이름의 컬렉션만 지우고 다시 만든다."""
    if client.collection_exists(name):
        if not recreate:
            existing = client.get_collection(name).config.params.vectors
            size = existing[DENSE].size if isinstance(existing, dict) and DENSE in existing else None
            if size != dense_size:
                raise ValueError(
                    f"컬렉션 '{name}' 의 dense 차원({size})이 임베딩 차원({dense_size})과 다르다. "
                    "임베딩 모델을 바꿨다면 recreate=True 로 다시 만들어라."
                )
            return
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: models.VectorParams(size=dense_size, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in ("article", "level", "parent_id"):
        client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)


def index_chunks(
    chunks: list[Chunk],
    client: QdrantClient | None = None,
    embeddings=None,
    name: str = COLLECTION_NAME,
    recreate: bool = False,
    batch_size: int = 32,
) -> int:
    """child 청크를 임베딩해 Qdrant 에 upsert 한다. 저장한 포인트 수를 반환한다.

    point id 가 chunk_id 에서 결정적으로 만들어지므로 다시 실행해도 중복되지 않는다.
    (다만 법령 개정으로 사라진 청크는 남으므로, 그럴 때는 recreate=True 를 쓴다.)
    """
    if not chunks:
        raise ValueError("인덱싱할 청크가 없다.")
    client = client or get_qdrant_client()

    # 임베딩을 먼저 끝내서, 실패해도 기존 컬렉션이 지워지지 않게 한다.
    dense_vectors = embed_texts([c.embed_text for c in chunks], embeddings, batch_size)
    ensure_collection(client, len(dense_vectors[0]), name, recreate)

    encoder = SparseEncoder()
    points = [
        models.PointStruct(
            id=c.point_id,
            vector={DENSE: dense, SPARSE: encoder.encode(c.embed_text)},
            payload=c.payload(),
        )
        for c, dense in zip(chunks, dense_vectors)
    ]
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=name, points=points[i : i + batch_size], wait=True)
    return len(points)
