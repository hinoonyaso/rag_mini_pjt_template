import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")

MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
TEMPERATURE = os.getenv("LLM_TEMPERATURE", 2)
MAX_TOKENS = os.getenv("LLM_MAX_TOKENS", 2086)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)


# RAG 파이프라인 (rag/)
# 노트북 테스트용 컬렉션(law_articles)과 분리하기 위해 별도 이름 사용
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "ai_basic_law")

BASE_DIR = Path(__file__).resolve().parent.parent
LAW_DATA_DIR = BASE_DIR / "data" / "ai_basic_law"
CANONICAL_JSON_PATH = BASE_DIR / "data" / "processed" / "ai_basic_law.canonical.json"

# 국가법령정보 공동활용 Open API (open.law.go.kr). OC 는 .env 의 LAW_API_OC 로만 둔다.
LAW_API_OC = os.getenv("LAW_API_OC")
LAW_API_URL = "https://www.law.go.kr/DRF/lawService.do"
LAW_MST = os.getenv("LAW_MST", "282791")  # 인공지능기본법 (법률 제21311호)
LAW_EFFECTIVE_DATE = os.getenv("LAW_EFFECTIVE_DATE", "20260721")  # 시행일 기준 버전(efYd)
LAW_RAW_XML_PATH = LAW_DATA_DIR / f"law_{LAW_MST}.xml"  # API 원본 XML 캐시

# 답변 거부 1차 기준: rerank 최고 점수(0~10)가 이 값 미만이면 답변 LLM 을 부르지 않고 거부한다.
# 기본값 5: 현재 골든셋에서 범위 밖 9개 중 3개를 거절하고 정답 질문 36개는 거절하지 않았다(8~10으로 올리면 5개). 주제가 법에 있는 함정 질문은 점수로 못 거른다.
# 빈 값(RAG_MIN_RELEVANCE=)이면 1차 거부를 끈다.
# 2차 거부(LLM 의 근거 없음 판단, 인용 검증 실패)는 이 값과 무관하게 항상 동작한다.
_min_rel = os.getenv("RAG_MIN_RELEVANCE", "5")
RAG_MIN_RELEVANCE = int(_min_rel) if _min_rel else None
