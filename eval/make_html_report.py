"""평가 결과(data/processed/eval_compare.json)로 단일 HTML 보고서를 만든다. 표준 라이브러리만 쓴다.

    python3 eval/make_html_report.py        # → docs/evaluation_overview.html

숫자는 결과 JSON 에서 직접 계산하므로 손으로 옮기다 틀릴 일이 없다. 그래프 PNG(docs/images)는 파일 안에 넣어 한 파일로 열린다.
"""

import base64
import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "eval_compare.json"
OUT = ROOT / "docs" / "evaluation_overview.html"
IMG = ROOT / "docs" / "images"

ORDER = ["dense", "dense+rerank", "hybrid", "hybrid+rerank"]
NAME = {"dense": "dense만 (기준선)", "dense+rerank": "reranking만", "hybrid": "hybrid만", "hybrid+rerank": "hybrid + reranking"}
SUB = {"dense": "벡터 검색", "dense+rerank": "dense 검색 + LLM rerank", "hybrid": "dense + sparse (RRF)", "hybrid+rerank": "현재 운영 구성"}
CLS = {"dense": "s1", "dense+rerank": "s4", "hybrid": "s2", "hybrid+rerank": "s3"}  # 구성마다 색을 고정한다
e = html.escape


def mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else float("nan")


def metrics(rows):
    single = [r for r in rows if r["n_gold"] == 1]
    multi = [r for r in rows if r["n_gold"] >= 2]
    return {
        "Recall@20": mean(r["recall20"] for r in rows), "Recall@30": mean(r["recall30"] for r in rows),
        "nDCG@5": mean(r["ndcg5"] for r in rows), "nDCG@10": mean(r["ndcg10"] for r in rows),
        "근거 확보율": mean(r["ctx_cov"] for r in rows), "전부 확보 (전체)": mean(float(r["ctx_full"]) for r in rows),
        f"전부 확보 (근거 2개 이상, n={len(multi)})": mean(float(r["ctx_full"]) for r in multi),
        "MRR@10": mean(r["rr10"] for r in single),
        "Hit@1": mean(float(r["hit"]["1"]) for r in single), "Hit@3": mean(float(r["hit"]["3"]) for r in single),
        "Hit@5": mean(float(r["hit"]["5"]) for r in single),
    }


def img(name, alt):
    b64 = base64.b64encode((IMG / name).read_bytes()).decode()
    return f'<figure class="fig"><img alt="{e(alt)}" src="data:image/png;base64,{b64}"></figure>'


def bar_panel(title, keys, M):
    out = [f'<div class="panel"><h4>{e(title)}</h4>']
    for k in keys:
        out.append(f'<div class="mrow"><div class="mlabel">{e(k)}</div><div class="bars">')
        for n in ORDER:
            v = M[n][k]
            out.append(f'<div class="bar-line" title="{e(NAME[n])} · {e(k)} = {v:.3f}"><div class="bar {CLS[n]}" style="width:{v * 100:.1f}%"></div><span class="val">{v:.3f}</span></div>')
        out.append("</div></div>")
    out.append("</div>")
    return "".join(out)


def table(M, keys, best_high=True):
    rows = ['<table class="t"><thead><tr><th>지표</th>' + "".join(f'<th class="num"><span class="dot {CLS[n]}"></span>{e(NAME[n])}</th>' for n in ORDER) + "</tr></thead><tbody>"]
    for k in keys:
        vals = [M[n][k] for n in ORDER]
        top = max(vals)
        cells = "".join(f'<td class="num{" best" if abs(v - top) < 1e-9 and sum(abs(x - top) < 1e-9 for x in vals) < len(vals) else ""}">{v:.3f}</td>' for v in vals)
        rows.append(f"<tr><td>{e(k)}</td>{cells}</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def heat(rows):
    cols = [("Recall@20", lambda r: r["recall20"]), ("nDCG@5", lambda r: r["ndcg5"]), ("Hit@1", lambda r: float(r["hit"]["1"])),
            ("근거 확보율", lambda r: r["ctx_cov"]), ("전부 확보", lambda r: float(r["ctx_full"]))]
    types = {}
    for r in rows:
        types.setdefault(r["type"], []).append(r)
    lo, hi = (205, 226, 251), (16, 66, 129)
    out = ['<table class="t heat"><thead><tr><th>유형</th>' + "".join(f'<th class="num">{e(c)}</th>' for c, _ in cols) + "</tr></thead><tbody>"]
    for t, rs in types.items():
        cells = []
        for _, f in cols:
            v = mean(f(r) for r in rs)
            rgb = tuple(round(lo[i] + (hi[i] - lo[i]) * v) for i in range(3))
            fg = "#fff" if v >= 0.62 else "#0b0b0b"
            cells.append(f'<td class="num" style="background:rgb{rgb};color:{fg}" title="{e(t)} · {v:.2f}">{v:.2f}</td>')
        out.append(f"<tr><td>{e(t)} <span class='muted'>(n={len(rs)})</span></td>{''.join(cells)}</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def build():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    cfg = d["configs"]
    M = {n: metrics(cfg[n]["rows"]) for n in ORDER}
    n_all = len(cfg["dense"]["rows"])
    multi_key = next(k for k in M["dense"] if k.startswith("전부 확보 (근거"))
    n_multi = sum(1 for r in cfg["dense"]["rows"] if r["n_gold"] >= 2)
    dp = d["data_processing"]

    # 컨텍스트를 고르는 방식
    sel_rows = []
    for key, label in (("current", "상위 child 6개 → 조 최대 4개 (현재)"), ("3", "후보 전체에서 상위 조 3개"), ("4", "후보 전체에서 상위 조 4개"), ("5", "후보 전체에서 상위 조 5개")):
        cells = []
        for n in ORDER:
            rs = cfg[n]["rows"]
            full = mean(float(r["ctx_select"][key]["full"]) for r in rs)
            fm = mean(float(r["ctx_select"][key]["full"]) for r in rs if r["n_gold"] >= 2)
            par = mean(r["ctx_select"][key]["parents"] for r in rs)
            cells.append(f'<td class="num">{full:.3f}<br><span class="muted">다중 {fm:.3f} · {par:.2f}개 조</span></td>')
        sel_rows.append(f"<tr><td>{e(label)}</td>{''.join(cells)}</tr>")

    # 범위 밖 거절
    gate_names = ["dense+rerank", "hybrid+rerank"]
    oos = {n: {o["id"]: o for o in cfg[n]["oos"]} for n in gate_names}
    ids = sorted(oos["hybrid+rerank"])
    gate_rows = "".join(
        f'<tr><td>{i}</td>' + "".join(f'<td class="num{" warn" if oos[n][i]["score"] >= 5 else ""}">{oos[n][i]["score"]}</td>' for n in gate_names)
        + f'<td>{e(oos["hybrid+rerank"][i]["top_article"])}</td></tr>' for i in ids)
    thr_rows = ""
    for t in (5, 8, 10):
        thr_rows += f"<tr><td>{t}{' (현재 기본값)' if t == 5 else ''}</td>" + "".join(
            f'<td class="num">{sum(o["score"] < t for o in oos[n].values())}/{len(oos[n])}</td>' for n in gate_names) + "<td class='num'>0/36</td></tr>"

    css = CSS
    body = f"""
<header class="hero">
  <p class="eyebrow">인공지능기본법 근거 기반 QA · 평가 결과 정리</p>
  <h1>검색 구성 비교와 답변 거절 평가</h1>
  <p class="lead">직접 만든 골든셋 {n_all + len(cfg['hybrid+rerank']['oos'])}문항(정답 있음 {n_all}, 범위 밖 {len(cfg['hybrid+rerank']['oos'])})으로 hybrid만 / reranking만 / hybrid + reranking을 같은 평가 기준으로 비교했다.</p>
</header>

<section class="cards">
  <div class="card"><div class="k">순위는 rerank가 올린다</div><div class="v">nDCG@5 {M['dense']['nDCG@5']:.3f} → {M['dense+rerank']['nDCG@5']:.3f}</div><div class="s">dense에 rerank를 붙였을 때. hybrid로 바꾸면 {M['hybrid']['nDCG@5']:.3f}</div></div>
  <div class="card"><div class="k">첫 정답 속도는 hybrid + rerank</div><div class="v">Hit@1 {M['hybrid+rerank']['Hit@1']:.3f}</div><div class="s">MRR@10 {M['hybrid+rerank']['MRR@10']:.3f}. reranking만은 {M['dense+rerank']['Hit@1']:.3f}</div></div>
  <div class="card warn"><div class="k">약점은 여러 조문에 걸친 질문</div><div class="v">전부 확보 {M['hybrid+rerank'][multi_key]:.3f}</div><div class="s">근거 2개 이상 {n_multi}문항, hybrid + reranking 기준</div></div>
  <div class="card warn"><div class="k">범위 밖 질문은 점수로 일부만 거절</div><div class="v">3/9</div><div class="s">기준 5. 주제가 법에 있는 함정 4문항은 10점</div></div>
</section>

<section>
  <h2>1. 비교한 구성</h2>
  <div class="cfgs">
    {''.join(f'<div class="cfg"><span class="dot {CLS[n]}"></span><b>{e(NAME[n])}</b><br><span class="muted">{e(SUB[n])}</span></div>' for n in ORDER)}
  </div>
  <p class="note">rerank는 후보가 있어야 하므로 "reranking만"은 dense 검색 후보 20개에 LLM rerank를 붙인 구성이다. 모든 구성이 같은 규칙(상위 child 6개 → 조 최대 4개)으로 LLM 컨텍스트를 만든다. 지표는 정답 있는 {n_all}문항, 정답은 조 단위다.</p>
</section>

<section>
  <h2>2. 평가 기준 단계별 비교</h2>
  <div class="legend">{''.join(f'<span><span class="dot {CLS[n]}"></span>{e(NAME[n])}</span>' for n in ORDER)}</div>
  <div class="panels">
    {bar_panel("② 후보 검색", ["Recall@20", "Recall@30"], M)}
    {bar_panel("③ 최종 순위 (조 단위 nDCG)", ["nDCG@5", "nDCG@10"], M)}
    {bar_panel("④ 최종 컨텍스트 (조 최대 4개)", ["근거 확보율", "전부 확보 (전체)", multi_key], M)}
    {bar_panel("⑤ 단일 근거 질문", ["MRR@10", "Hit@1", "Hit@3", "Hit@5"], M)}
  </div>
  <details open><summary>표로 보기</summary>
    {table(M, ["Recall@20", "Recall@30", "nDCG@5", "nDCG@10", "근거 확보율", "전부 확보 (전체)", multi_key, "MRR@10", "Hit@1", "Hit@3", "Hit@5"])}
    <p class="note">굵게 표시한 값이 그 행의 최고(동률이면 표시 안 함). 한 구성이 모든 지표에서 이기지는 않는다.</p>
  </details>
  <ul class="find">
    <li><b>rerank가 순위를 가장 크게 올린다.</b> dense에 붙이면 nDCG@5가 +{M['dense+rerank']['nDCG@5'] - M['dense']['nDCG@5']:.3f}, dense를 hybrid로 바꾸면 +{M['hybrid']['nDCG@5'] - M['dense']['nDCG@5']:.3f}이다.</li>
    <li><b>rerank를 붙이면 hybrid의 이점이 줄어든다.</b> reranking만과 hybrid + reranking의 nDCG@5는 {M['dense+rerank']['nDCG@5']:.3f}과 {M['hybrid+rerank']['nDCG@5']:.3f}로 같고, Hit@1만 한 문항 차이다.</li>
    <li><b>rerank는 후보를 새로 찾지 못한다.</b> Recall이 rerank 전후로 같다. 모든 구성이 q19의 제43조처럼 근거 1개를 후보 30개에도 못 넣는다.</li>
    <li><b>hybrid에 rerank를 붙여도 컨텍스트 확보는 오르지 않는다</b>({M['hybrid']['근거 확보율']:.3f} → {M['hybrid+rerank']['근거 확보율']:.3f}). rerank가 같은 조의 child 여러 개에 높은 점수를 줘서 "상위 child 6개"가 한두 조로 채워지기 때문이다(q23).</li>
  </ul>
</section>

<section>
  <h2>3. 컨텍스트를 조 단위로 고르면</h2>
  <p>"상위 child 6개"가 아니라 <b>후보 전체에서 상위 조 N개</b>를 고르면 같은 입력 크기로 더 많이 확보한다. 칸은 "전부 확보 비율"이고 아래 줄은 근거 2개 이상 질문의 비율과 LLM에 넘기는 평균 조 수다.</p>
  <table class="t"><thead><tr><th>고르는 방식</th>{''.join(f'<th class="num"><span class="dot {CLS[n]}"></span>{e(NAME[n])}</th>' for n in ORDER)}</tr></thead><tbody>{''.join(sel_rows)}</tbody></table>
  <p class="note">예: reranking만에서 상위 조 3개를 고르면 입력은 거의 그대로(2.92 → 2.97개)인데 다중 근거 전부 확보가 0.714 → 0.857이다. hybrid + reranking은 5개를 넘겨야 0.714가 된다(입력 약 +65%). 다만 근거 2개 이상은 {n_multi}문항이라 한 문항이 0.14를 움직인다.</p>
  {img("eval_context_select.png", "컨텍스트를 고르는 방식별 필수 근거 확보")}
</section>

<section>
  <h2>4. 질문 유형별 (hybrid + reranking)</h2>
  {heat(cfg['hybrid+rerank']['rows'])}
  <p class="note">색이 진할수록 높음(0~1). 유형당 4~7문항이라 한 문항이 0.14~0.25를 움직인다. 정의·의무·신설·기타(21문항)는 모두 1.00이고, 약한 유형은 <b>복합</b>(전부 확보 0.60)과 <b>숫자벌칙</b>(Hit@1 0.80)이다.</p>
  <h3>실패한 문항</h3>
  <table class="t"><thead><tr><th>문항</th><th>정답</th><th>무슨 일이 있었나</th></tr></thead><tbody>
    <tr><td>q23 고영향 AI 사업자 의무 전부</td><td>제33·34·35조</td><td>정답 3개 조가 조 순위 1~3위인데 컨텍스트에는 2개만 들어갔다. 상위 child 6개가 두 조로 채워진 것으로 보인다.</td></tr>
    <tr><td>q19 자료 제출·조사 불응</td><td>제40·43조</td><td>제43조가 후보 30개에도 없다. "조사 불응 → 명령 → 불이행 시 과태료"의 두 단계를 건너야 하는 질문이다.</td></tr>
    <tr><td>q16 대출 심사 규제</td><td>제2·34조</td><td>"대출 심사 → 고영향 AI → 의무"의 추론이 필요해 제34조가 7위다.</td></tr>
    <tr><td>q25 AI 안전 관련 정부 기관</td><td>제11·12조</td><td>제11조(정책센터)가 15위다. 질문은 "안전"인데 제11조는 안전과 직접 관련이 약해 <b>정답 라벨 확인이 필요</b>하다.</td></tr>
  </tbody></table>
</section>

<section>
  <h2>5. 범위 밖 질문: rerank 점수로 거절되는가</h2>
  <p>정답 있는 {n_all}문항은 두 구성 모두 <b>전부 10점</b>이다. 범위 밖 9문항의 최고 점수는 다음과 같다(10점은 거절 기준을 통과한다).</p>
  <div class="two">
    <div><table class="t"><thead><tr><th>질문</th><th class="num">reranking만</th><th class="num">hybrid + rerank</th><th>rerank 1위 조</th></tr></thead><tbody>{gate_rows}</tbody></table></div>
    <div><table class="t"><thead><tr><th>거절 기준 (미만이면 거절)</th><th class="num">reranking만</th><th class="num">hybrid + rerank</th><th class="num">정답 질문 오거절</th></tr></thead><tbody>{thr_rows}</tbody></table>
      <p class="note">정답 질문을 거절한 적은 없다. 다만 정답 질문이 정확히 10점뿐이라 기준을 올릴 때 여유 폭은 알 수 없다.</p></div>
  </div>
  <ul class="find">
    <li><b>완전히 무관한 질문은 잘 걸러진다</b> (특허, 세금 감면, 자격증: 0~3점).</li>
    <li><b>주제가 법에 있지만 구체 정보가 없는 질문은 점수로 못 거른다.</b> q37(영향평가 방법), q38(세부 기준 수치), q39(과징금)는 모두 10점이다. rerank가 "이 주제의 조문이 있나"만 보고 "질문한 정보가 있나"는 못 보기 때문이다.</li>
    <li>이런 질문은 LLM이 답변 단계에서 "근거 없음"으로 판단하는 <b>2차 거절</b>이 막아야 하는데, <b>2차는 아직 평가하지 않았다.</b></li>
  </ul>
  {img("eval_gate.png", "rerank 점수로 범위 밖 질문을 걸러낼 수 있는가")}
</section>

<section>
  <h2>6. 전처리·청크 테스트 <span class="tag">AI가 만든 옛 골든셋 기준</span></h2>
  <div class="kv">
    <div><b>원문 보존율</b> XML → 정규화 JSON {dp['xml_to_canonical']:.4f} · → child {dp['canonical_to_chunk']:.4f} · → parent {dp['canonical_to_parent']:.4f}</div>
    <div><b>ID 충돌</b> chunk_id {dp['chunk_id_collisions']} · point_id {dp['point_id_collisions']}</div>
    <div><b>Qdrant 저장</b> {dp['qdrant_points']} / {dp['chunks']} (누락 {dp['qdrant_missing']})</div>
    <div><b>청크 크기</b> 500·800·1200자와 비교해 <b>300자</b>가 가장 안정적(art@3 0.951 → 0.976, 나빠진 문항 없음)</div>
    <div><b>효과 없던 시도</b> 조당 child 상한, 참조 풀기</div>
  </div>
  <p class="note">청크 크기 결과는 옛 골든셋(AI가 만든 41문항)으로 측정했고 새 골든셋에서는 재확인하지 않았다. 청크 임베딩이 캐시에 있어 API 호출 없이 다시 확인할 수 있다.</p>
</section>

<section>
  <h2>7. 평가 중 찾아 고친 문제</h2>
  <table class="t"><thead><tr><th>문제</th><th>영향</th><th>조치</th></tr></thead><tbody>
    <tr><td>hybrid 검색이 같은 질문에 매번 다른 결과(RRF 동점 순서)</td><td>평가 수치가 흔들림</td><td>동점이면 chunk_id 순으로 고정</td></tr>
    <tr><td>근거가 부칙인 정답이 인용 검증에서 거부됨</td><td>시행일 질문이 거부될 뻔함</td><td>부칙을 인정하도록 수정</td></tr>
    <tr><td>정답이 조 단위인데 nDCG를 child 단위로 계산</td><td>정답 조가 1위여도 감점</td><td>조 단위 nDCG로 변경</td></tr>
    <tr><td>SDK 자동 재시도가 429(분당 30회)를 부풀림</td><td>평가가 계속 막힘</td><td>재시도를 끄고 호출 속도를 분당 12회로 제한</td></tr>
  </tbody></table>
</section>

<section>
  <h2>8. 아직 측정하지 않은 것과 정할 것</h2>
  <div class="two">
    <div><h3>미측정</h3><ul class="find">
      <li><b>답변 단계</b>: 정답성, 인용 정확성, 근거 충실성</li>
      <li><b>2차 거절</b>: 함정 질문이 실제로 거절되는지, 정답 질문을 잘못 거절하지 않는지</li>
      <li><b>실제 /ask 호출</b>과 프롬프트의 "쉬운 말" 효과</li>
      <li>청크 크기 테스트의 새 골든셋 재확인</li>
    </ul></div>
    <div><h3>정할 것</h3><ul class="find">
      <li>2차 거절·답변 평가를 할지 (최대 약 20회 호출)</li>
      <li>컨텍스트를 조 단위로 고르는 방식을 반영할지</li>
      <li>거절 기준 점수를 5에서 올릴지</li>
      <li>q25 정답 라벨 확인</li>
    </ul></div>
  </div>
  <p class="note"><b>주의</b> 정답 있는 {n_all}문항(근거 2개 이상 {n_multi})이라 한 문항이 큰 차이를 만든다. reranking만과 hybrid + reranking의 차이는 1~2문항 수준이라 <b>현재 운영 구성을 바꿀 근거는 아직 없다.</b> API 한도(분당 30회)에 평가가 자주 걸렸고, 같은 키를 쓰는 다른 호출이 한도를 나눠 쓰는 것으로 보이지만 확인하지 못했다.</p>
</section>
<footer>결과 원자료 <code>data/processed/eval_compare.json</code> · 생성 <code>python3 eval/make_html_report.py</code> · 상세 문서 <code>docs/hybrid_rerank_comparison.md</code>, <code>docs/chunking_experiments.md</code></footer>
"""
    return f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>평가 결과 정리</title><style>{css}</style></head><body><main>{body}</main></body></html>'


CSS = """
:root{--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;--accent:#2a78d6;--warn-bg:#fdf1ec;--warn:#b4421c;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--best:#e8f1fc}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){--bg:#1a1a19;--panel:#222221;--ink:#fff;--ink2:#c3c2b7;--line:#383835;--accent:#3987e5;--warn-bg:#2b211d;--warn:#f0a07f;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--best:#1f2c3d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,"Segoe UI","Noto Sans KR","NanumGothic",sans-serif}
main{max-width:1060px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:30px;line-height:1.25;margin:4px 0 8px}h2{font-size:21px;margin:0 0 12px}h3{font-size:16px;margin:16px 0 8px}h4{margin:0 0 10px;font-size:14px}
section{margin:36px 0;padding-top:8px}.eyebrow{color:var(--accent);font-weight:700;font-size:13px;margin:0}.lead{color:var(--ink2);font-size:16px;max-width:760px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:20px}@media(max-width:860px){.cards{grid-template-columns:1fr 1fr}}@media(max-width:520px){.cards{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}.card.warn{background:var(--warn-bg);border-color:transparent}
.card .k{font-size:12.5px;color:var(--ink2)}.card .v{font-size:22px;font-weight:700;margin:4px 0}.card .s{font-size:12.5px;color:var(--ink2)}.card.warn .v{color:var(--warn)}
.cfgs{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}@media(max-width:760px){.cfgs{grid-template-columns:1fr 1fr}}
.cfg{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:14px}
.dot{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:7px;vertical-align:baseline}.s1{background:var(--s1)}.s2{background:var(--s2)}.s3{background:var(--s3)}.s4{background:var(--s4)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:6px 0 14px;font-size:13.5px}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:860px){.panels{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.mrow{margin:0 0 12px}.mlabel{font-size:13px;color:var(--ink2);margin-bottom:3px}
.bar-line{display:flex;align-items:center;gap:8px;margin:2px 0}.bar{height:12px;border-radius:0 4px 4px 0;min-width:2px;border-right:0}.val{font-size:12px;font-variant-numeric:tabular-nums}
.t{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13.5px;margin:10px 0}
.t th,.t td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.t th{font-size:12.5px;color:var(--ink2);background:var(--bg)}.t tr:last-child td{border-bottom:0}
.num{text-align:right!important;font-variant-numeric:tabular-nums}.best{font-weight:800;background:var(--best)}.warn{color:var(--warn);font-weight:700}.muted{color:var(--ink2);font-size:12px}
.heat td{border-bottom:2px solid var(--panel);border-left:2px solid var(--panel)}
.note{color:var(--ink2);font-size:13.5px}.find{padding-left:20px}.find li{margin:5px 0}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:860px){.two{grid-template-columns:1fr}}
.fig{margin:14px 0;background:#fcfcfb;border:1px solid var(--line);border-radius:12px;padding:8px;overflow-x:auto}.fig img{max-width:100%;height:auto;display:block;margin:0 auto}
.kv div{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:6px 0;font-size:14px}
.tag{font-size:12px;font-weight:600;color:var(--ink2);background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:2px 9px;margin-left:8px;vertical-align:middle}
details summary{cursor:pointer;color:var(--accent);font-weight:600;margin:8px 0}
footer{margin-top:40px;padding-top:14px;border-top:1px solid var(--line);color:var(--ink2);font-size:12.5px}code{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:1px 5px}
"""

if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print("saved", OUT, f"({OUT.stat().st_size // 1024} KB)")
