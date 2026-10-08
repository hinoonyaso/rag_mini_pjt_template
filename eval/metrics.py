"""평가 지표 계산 (순수 함수: API·DB 없이 테스트할 수 있다).

정답은 골든셋의 gold = [{"article": "제7조", "paragraph": 8}, ...] 이다. paragraph 가 없으면 그 조 어디든 맞는 근거다.
child payload 와 gold 한 건의 관계를 등급으로 나눈다.
    2  정답 근거를 담고 있다 (같은 항이거나, 항 지정이 없거나, 조 전체를 담은 child)
    1  같은 조이지만 다른 항이다 (부분 관련)
    0  무관
"""

import math
import re

from rag.chunker import CIRCLED_TO_INT

INT_TO_CIRCLED = {v: k for k, v in CIRCLED_TO_INT.items()}
_RE_PARA = re.compile(r"제(\d+)항")


def grade(payload: dict, gold: dict) -> int:
    if payload["article"] != gold["article"]:
        return 0
    if "paragraph" not in gold:
        return 2
    if payload.get("paragraph") == INT_TO_CIRCLED[gold["paragraph"]] or payload["level"] == "article":
        return 2
    return 1


def best_grade(payload: dict, golds: list[dict]) -> int:
    return max((grade(payload, g) for g in golds), default=0)


def recall_at_k(ranked: list[dict], golds: list[dict], k: int, strict: bool = True) -> float:
    """필수 근거(gold) 중 상위 k 후보 안에서 확보된 비율. strict=True 는 항까지, False 는 조까지 맞아야 한다."""
    top = ranked[:k]
    found = 0
    for g in golds:
        if strict:
            found += any(grade(p, g) == 2 for p in top)
        else:
            found += any(p["article"] == g["article"] for p in top)
    return found / len(golds)


def _dcg(grades: list[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: list[dict], golds: list[dict], corpus: list[dict], k: int) -> float:
    """등급(2/1/0)을 쓴 nDCG@k. 이상적인 순서는 코퍼스 전체 child 의 등급을 내림차순으로 놓은 것이다."""
    actual = _dcg([best_grade(p, golds) for p in ranked[:k]])
    ideal = _dcg(sorted((best_grade(p, golds) for p in corpus), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def ndcg_articles(ranked: list[dict], golds: list[dict], k: int) -> float:
    """조 단위 nDCG@k. 정답이 조 단위일 때 쓴다.

    child 단위로 계산하면 제2조의 child 22개가 모두 정답이 되어, 정답 조가 1위여도 "같은 조 child 가 5개 안 나왔다"는 이유로 감점된다.
    LLM 에는 조 단위로 넘기므로, 같은 조는 처음 나온 순서 하나로 합친 순위에서 계산한다 (정답 조=1, 그 외 0).
    """
    arts = list(dict.fromkeys(p["article"] for p in ranked))[:k]
    gold_arts = {g["article"] for g in golds}
    actual = _dcg([1 if a in gold_arts else 0 for a in arts])
    ideal = _dcg([1] * min(len(gold_arts), k))
    return actual / ideal if ideal > 0 else 0.0


def reciprocal_rank(ranked: list[dict], golds: list[dict], k: int = 10) -> float:
    for i, p in enumerate(ranked[:k], 1):
        if best_grade(p, golds) == 2:
            return 1 / i
    return 0.0


def hit_at_k(ranked: list[dict], golds: list[dict], k: int) -> bool:
    return any(best_grade(p, golds) == 2 for p in ranked[:k])


def context_coverage(context_articles: set[str], golds: list[dict]) -> float:
    """LLM 에 넘긴 조(parent) 안에 정답 조가 있는 비율. 조 전체를 넘기므로 항 지정은 자동으로 포함된다."""
    return sum(g["article"] in context_articles for g in golds) / len(golds)


def citation_scores(citations: list[str], golds: list[dict]) -> tuple[float, float]:
    """(정밀도, 재현율). 인용 "제7조제8항" 이 gold {제7조, 8항} 에 맞으면 정확한 인용이다. 항이 없는 인용은 조 단위로만 본다."""

    def cited(cite: str, g: dict) -> bool:
        m = re.match(r"(제\d+조(?:의\d+)?|부칙)", cite.strip())
        if not m or m.group(1) != g["article"]:
            return False
        para = _RE_PARA.search(cite)
        return "paragraph" not in g or para is None or int(para.group(1)) == g["paragraph"]

    if not citations:
        return 0.0, 0.0
    precision = sum(any(cited(c, g) for g in golds) for c in citations) / len(citations)
    recall = sum(any(cited(c, g) for c in citations) for g in golds) / len(golds)
    return precision, recall


def refusal_precision_recall(records: list[tuple[bool, bool]]) -> tuple[float, float]:
    """records = [(답변 불가능 질문인가, 거절했는가), ...]. 정밀도 = 거절 중 실제 불가능 비율, 재현율 = 불가능 질문 중 거절 비율."""
    refused = [r for r in records if r[1]]
    unanswerable = [r for r in records if r[0]]
    precision = sum(r[0] for r in refused) / len(refused) if refused else 1.0
    recall = sum(r[1] for r in unanswerable) / len(unanswerable) if unanswerable else 1.0
    return precision, recall


def _self_test() -> None:
    P = lambda art, para=None, level="paragraph": {"article": art, "paragraph": para, "level": level}
    gold = [{"article": "제7조", "paragraph": 8}]
    assert grade(P("제7조", "⑧"), gold[0]) == 2
    assert grade(P("제7조", "⑦"), gold[0]) == 1
    assert grade(P("제7조", None, "article"), gold[0]) == 2
    assert grade(P("제8조", "⑧"), gold[0]) == 0
    assert grade(P("제2조", None, "item"), {"article": "제2조"}) == 2  # 항 지정 없으면 그 조의 모든 child 가 정답
    ranked = [P("제8조", "①"), P("제7조", "⑦"), P("제7조", "⑧")]  # 정답은 3위
    assert recall_at_k(ranked, gold, 2) == 0 and recall_at_k(ranked, gold, 3) == 1
    assert recall_at_k(ranked, gold, 2, strict=False) == 1  # 조 단위로는 2위에서 확보
    corpus = ranked + [P("제9조", "①")]
    perfect = [P("제7조", "⑧"), P("제7조", "⑦"), P("제8조", "①")]
    assert abs(ndcg_at_k(perfect, gold, corpus, 3) - 1.0) < 1e-9  # 이상적 순서
    assert 0 < ndcg_at_k(ranked, gold, corpus, 3) < 1
    assert ndcg_articles([P("제9조"), P("제9조"), P("제7조")], [{"article": "제7조"}], 5) == _dcg([0, 1]) / _dcg([1])  # 같은 조는 하나로 합쳐 2위
    assert ndcg_articles([P("제2조"), P("제3조")], [{"article": "제2조"}], 5) == 1.0  # 정답 조 1위면 만점
    assert reciprocal_rank(ranked, gold) == 1 / 3 and hit_at_k(ranked, gold, 3) and not hit_at_k(ranked, gold, 2)
    assert context_coverage({"제7조"}, [{"article": "제7조"}, {"article": "제43조"}]) == 0.5
    assert citation_scores(["제7조제8항", "제9조"], gold) == (0.5, 1.0)
    assert citation_scores(["제7조제7항"], gold) == (0.0, 0.0)  # 같은 조라도 다른 항을 인용하면 틀린 인용
    assert refusal_precision_recall([(True, True), (True, False), (False, False), (False, True)]) == (0.5, 0.5)


if __name__ == "__main__":
    _self_test()
    print("metrics 자체 검증 통과")
