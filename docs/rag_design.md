# RAG 설계 결정 기록

대상: 인공지능기본법 (법률 제21311호, MST 282791, 시행 2026-07-21 기준, 46개 조문). 코드는 `rag/` 참고.
입력: 국가법령정보 Open API `lawService.do?target=law&type=XML` (OC 는 `.env` 의 `LAW_API_OC`).

## 파이프라인

```
법령 API(XML, 원본은 data/ai_basic_law/law_282791.xml 에 캐시) → 태그 → Canonical JSON(data/processed/)
 → Parent-Child 청킹 → Embedding(dense) + sparse → Qdrant(ai_basic_law)
 → Hybrid(RRF) 검색 → LLM Rerank → 조(parent) 컨텍스트 → LLM 구조화 답변
```

실행: `uv run python -m rag.pipeline parse [--refresh] | ingest [--refresh] [--recreate] | ask "질문"`

## 결정과 근거

| 결정 | 이유 / 근거 | 절충점 |
|---|---|---|
| 입력을 HWPX 대신 법령 API(XML)로 | 응답에 조·항·호·목이 태그(`조문단위/항/호/목`)로 나뉘어 있어 접두어 정규식 추측이 필요 없음. 가지조문·가지호는 `조문가지번호`·`호가지번호` 필드, 장·절은 `조문여부=전문` 단위. 부칙도 3건 전부 제공(HWPX는 최신 1건) | OC 발급·서버 IP 등록 필요. 인증 오류는 HTTP 200 + 오류 문서로 오므로 루트 태그가 `법령`인지로 판별. 원본 XML을 캐시해 재현 가능 |
| 번호 없는 항(제2조·제29조)의 호는 조 직속 호로 | 실제 XML에서 `항번호` 없는 `항`이 2개이며 그 안에 호가 있음 | — |
| 파싱 검증 | 조 번호 단조 증가·빈 조문 검사. 구 HWPX 원본과 교차 대조(아래) | HWPX 파서는 제거함. 대조용 원본 파일은 data/ 에 남아 있음 |
| Child 크기 기준 300자, 항→호→목 순으로 분할 | 골든셋 41문항으로 500·800·1200자와 비교해 300자가 가장 안정적(art@3 0.951→0.976, mrr 0.911→0.929, 나빠진 문항 없음). 도입부를 child에 붙여 단독으로 뜻이 통함. 상세: [chunking_experiments.md](chunking_experiments.md) | child 271개(중앙값 123자, 최대 378자). 문장 중간은 자르지 않아 300자 초과가 4개 있음. 300자 미만은 미시도 |
| Parent 원문을 child payload에 저장 | 조 최대 수 KB, 총 271개라 중복 비용이 작고 별도 조회 저장소가 필요 없음 | 법령이 수천 조 규모가 되면 parent를 별도 컬렉션으로 분리 검토 |
| point id = `uuid5(chunk_id)` | 재적재해도 중복되지 않음 | 개정으로 사라진 청크는 남음 → `--recreate` 사용 |
| sparse = 글자 bigram + 조문 참조 토큰, Qdrant IDF | 한국어 형태소 분석기(추가 의존성) 없이 조사 변형(`인공지능을/이`)과 `제6조` 같은 정확 매칭을 처리 | 형태소 분석 대비 노이즈가 있음. 품질이 부족하면 kiwipiepy 도입을 평가 |
| Rerank = 기존 LLM(listwise, function_calling) | 추가 모델·GPU 불필요. 라우터가 function_calling에서만 구조화 출력이 동작함(notebook으로 확인) | 호출 1회 지연·비용 증가. 실패 시 검색 순서로 폴백. 효과는 현재 골든셋으로 평가 중 |
| 답변 거부 두 겹: ① rerank 최고 점수가 기준 미만이면 답변 LLM 호출 없이 거부 ② LLM이 근거 없음이라 하거나 인용 검증에 실패하면 거부 | 법령에 없는 질문에 그럴듯하게 답하지 않기 위한 장치 | **기준 점수(기본 5)의 1차 거절은 현재 골든셋에서 범위 밖 3/9만 거절하고(오거절 0/36), 주제가 법에 있는 함정 질문은 점수로 못 거른다. 2차는 미평가.** ([hybrid_rerank_comparison.md](hybrid_rerank_comparison.md) 3-5장) 부칙을 인용하는 답변이 검증에서 걸리던 버그는 수정함 |
| 컬렉션 `ai_basic_law` 별도 사용 | 노트북 테스트용 `law_articles`(unnamed dense 벡터)와 스키마가 달라 덮어쓰면 안 됨 | — |
| 답변 인용 검증 | 모델이 낸 `cited_articles`가 제공 컨텍스트의 조에 있는지 확인, 없으면 `unverified_citations`로 분리 | 항·호 수준 환각은 조 단위로만 검증 |

## 확인한 동작 (2026-10-08, 로컬 Qdrant 1.19.2)

- 파싱(XML): 조/항/호/목 = 46/164/191/13. 독립 소스인 HWPX 본문 373줄과 대조해 누락 0줄, 개수도 일치. 가지조문 3개·가지호 8개·부칙 3건 정상.
- 적재: 271 child 저장(부칙 3건 포함, 기준 300자).

## 알려진 제한 / 미확인

- 검색·rerank 비교(hybrid만 / reranking만 / hybrid+reranking)는 [hybrid_rerank_comparison.md](hybrid_rerank_comparison.md) 참고(정답 있는 36문항). 답변 거부와 답변 단계(정답성·인용·충실성)는 아직 평가하지 않았다. 300자 미만 청크 크기는 미측정.
- 부칙 chunk(3건)가 일부 질의의 후보에 섞일 수 있다. 필요하면 부칙을 검색에서 제외하는 필터를 검토.
- 시행령은 아직 포함하지 않음(MST 288781). 포함하려면 법령 간 인용 표기·메타데이터 구분 필요.
- 조문 간 참조(`제2조제1호에 따른`)는 해소하지 않음.
