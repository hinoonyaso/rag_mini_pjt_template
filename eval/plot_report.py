"""평가 결과(data/processed/eval_compare.json)를 그래프(PNG)로 만든다. docs/images/ 에 저장한다.

프로젝트 의존성이 아니라서 별도로 설치된 matplotlib 이 있는 파이썬으로 실행한다 (API·DB·프로젝트 모듈을 쓰지 않는다).

    python3 eval/plot_report.py

결과 JSON 은 `uv run python -m eval.evaluate --report` 가 만든다.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "eval_compare.json"
OUT = ROOT / "docs" / "images"

# 색상 토큰 (dataviz 기본 팔레트, light). 범주형 3색은 validate_palette.js 로 검증했다 (CVD·정상시력 기준 통과).
# 청록은 대비 3:1 미만이라 막대마다 값을 직접 표기한다(relief rule).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# 엔티티(구성)에 색을 고정한다. 4색은 validate_palette.js 로 검증했고(인접쌍 CVD·정상시력 통과), 청록·노랑은 대비가 낮아 값 라벨로 보완한다.
COLOR = {"dense": "#2a78d6", "hybrid": "#eb6834", "hybrid+rerank": "#1baf7a", "dense+rerank": "#eda100"}
LABEL = {"dense": "dense만 (기준선)", "dense+rerank": "reranking만 (dense 검색 + LLM rerank)", "hybrid": "hybrid만 (dense+sparse, RRF)",
         "hybrid+rerank": "hybrid + reranking (현재 운영 구성)"}
ORDER = ("dense", "dense+rerank", "hybrid", "hybrid+rerank")

plt.rcParams.update({
    "font.family": "NanumGothic", "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
})


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def stage_metrics(rows):
    single = [r for r in rows if r["n_gold"] == 1]
    multi = [r for r in rows if r["n_gold"] >= 2]
    return {
        "Recall@20": mean(r["recall20"] for r in rows), "Recall@30": mean(r["recall30"] for r in rows),
        "nDCG@5": mean(r["ndcg5"] for r in rows), "nDCG@10": mean(r["ndcg10"] for r in rows),
        "근거 확보율": mean(r["ctx_cov"] for r in rows),
        "전부 확보(전체)": mean(float(r["ctx_full"]) for r in rows),
        f"전부 확보(근거 2개↑, n={len(multi)})": mean(float(r["ctx_full"]) for r in multi),
        "MRR@10": mean(r["rr10"] for r in single),
        "Hit@1": mean(float(r["hit"]["1"]) for r in single), "Hit@3": mean(float(r["hit"]["3"]) for r in single),
        "Hit@5": mean(float(r["hit"]["5"]) for r in single),
    }


def style_axis(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_xlim(0, 1.13)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_stages(data):
    names = [n for n in ORDER if n in data["configs"]]
    m = {n: stage_metrics(data["configs"][n]["rows"]) for n in names}
    n_all = len(data["configs"][names[0]]["rows"])
    multi_key = next(k for k in m[names[0]] if k.startswith("전부 확보(근거"))
    panels = [
        ("② 후보 검색", ["Recall@20", "Recall@30"]),
        ("③ 최종 순위 (조 단위 nDCG)", ["nDCG@5", "nDCG@10"]),
        ("④ 최종 컨텍스트 (조 최대 4개)", ["근거 확보율", "전부 확보(전체)", multi_key]),
        ("⑤ 단일 근거 질문", ["MRR@10", "Hit@1", "Hit@3", "Hit@5"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), gridspec_kw={"height_ratios": [2, 3.3]})
    h = 0.78 / len(names)
    for ax, (title, keys) in zip(axes.flat, panels):
        style_axis(ax)
        for i, key in enumerate(keys):
            for j, n in enumerate(names):
                y = i + (j - (len(names) - 1) / 2) * (h + 0.04)
                v = m[n][key]
                ax.barh(y, v, height=h, color=COLOR[n], edgecolor=SURFACE, linewidth=2)
                ax.text(v + 0.012, y, f"{v:.3f}", va="center", ha="left", fontsize=9.5, color=INK)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys, fontsize=10.5)
        ax.set_ylim(len(keys) - 0.4, -0.6)
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=8)
    fig.suptitle(f"평가 기준 단계별 비교 - 정답 있는 질문 {n_all}개", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.legend(handles=[Patch(facecolor=COLOR[n], label=LABEL[n]) for n in names], loc="upper left",
               bbox_to_anchor=(0.01, 0.945), ncol=2, frameon=False, fontsize=10.5)
    fig.text(0.01, 0.005, "값이 높을수록 좋음. 정답은 조 단위. ① 데이터 처리(원문 보존율 1.0, ID 충돌 0)는 그래프 대신 문서의 표 참고.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    return fig


def fig_types(data, name):
    rows = data["configs"][name]["rows"]
    types, order = {}, []
    for r in rows:
        if r["type"] not in types:
            order.append(r["type"])
        types.setdefault(r["type"], []).append(r)
    cols = [("Recall@20", lambda x: x["recall20"]), ("nDCG@5", lambda x: x["ndcg5"]), ("Hit@1", lambda x: float(x["hit"]["1"])),
            ("확보율", lambda x: x["ctx_cov"]), ("전부 확보", lambda x: float(x["ctx_full"]))]
    grid = [[mean(f(r) for r in types[t]) for _, f in cols] for t in order]
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"])
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(order) + 2.1))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11, color="#ffffff" if v >= 0.62 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c for c, _ in cols], fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{t} (n={len(types[t])})" for t in order], fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for k in range(len(order) + 1):  # 칸 사이 2px 표면색 간격
        ax.axhline(k - 0.5, color=SURFACE, linewidth=2)
    for k in range(len(cols) + 1):
        ax.axvline(k - 0.5, color=SURFACE, linewidth=2)
    fig.suptitle(f"질문 유형별 성능 - {name}", x=0.01, ha="left", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.01, 0.01, "색이 진할수록 높음 (0~1). 숫자는 유형 평균. 문항이 4~7개라 한 문항이 0.14~0.25를 움직인다.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


def fig_ctx(data):
    """LLM 에 넘기는 조를 고르는 방식별 근거 확보. 4개 구성을 한 그래프에서 비교한다 (전체 질문 | 근거가 2개 이상인 질문)."""
    names = [n for n in ORDER if n in data["configs"]]
    methods = [("current", "상위 child 6개에서\n조 최대 4개 (현재)"), ("3", "후보 전체에서\n상위 조 3개"), ("4", "후보 전체에서\n상위 조 4개"), ("5", "후보 전체에서\n상위 조 5개")]
    panels = [("전부 확보 - 전체 질문", lambda rows: [float(r["ctx_select"][k]["full"]) for r in rows]),
              ("전부 확보 - 근거가 2개 이상인 질문", lambda rows: [float(r["ctx_select"][k]["full"]) for r in rows if r["n_gold"] >= 2])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharey=True)
    h = 0.74 / len(names)
    for ax, (title, getter) in zip(axes, panels):
        style_axis(ax)
        for i, (k, _) in enumerate(methods):
            for j, n in enumerate(names):
                y = i + (j - (len(names) - 1) / 2) * (h + 0.03)
                vals = getter(data["configs"][n]["rows"])
                v = mean(vals)
                ax.barh(y, v, height=h, color=COLOR[n], edgecolor=SURFACE, linewidth=2)
                ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=9, color=INK)
        n_q = len(getter(data["configs"][names[0]]["rows"]))
        ax.set_title(f"{title} (n={n_q})", loc="left", fontsize=12, fontweight="bold", pad=8)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels([m for _, m in methods], fontsize=10)
    axes[0].set_ylim(len(methods) - 0.5, -0.5)
    fig.suptitle("LLM에 넘기는 조를 고르는 방식별 필수 근거 확보", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.legend(handles=[Patch(facecolor=COLOR[n], label=LABEL[n]) for n in names], loc="upper left",
               bbox_to_anchor=(0.01, 0.935), ncol=2, frameon=False, fontsize=10.5)
    fig.text(0.01, 0.005, "조를 많이 넘길수록 확보는 오르지만 LLM 입력(API 비용)도 는다. 평균 조 수: 현재 약 2.9~3.1개, 상위 3/4/5개는 각각 약 3.0/3.9/4.8개.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.025, 1, 0.88))
    return fig


def fig_gate(data):
    """rerank 최고 점수로 범위 밖 질문을 걸러낼 수 있는가: 정답 있는 질문 vs 범위 밖 질문의 점수 분포."""
    names = [n for n in ("dense+rerank", "hybrid+rerank") if n in data["configs"] and data["configs"][n].get("oos")]
    if not names:
        return None
    GRAY, VIOLET = "#8a8984", "#4a3aa7"
    fig, axes = plt.subplots(1, len(names), figsize=(6.6 * len(names), 5.4), sharey=True)
    axes = [axes] if len(names) == 1 else list(axes)
    for ax, n in zip(axes, names):
        ans = [r["score"] for r in data["configs"][n]["rows"] if r["score"] is not None]
        oos = data["configs"][n]["oos"]
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.6, 11.6)
        ax.set_xticks(range(0, 11))
        ax.axvline(5, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(5.1, 1.62, "거절 기준(기본 5)\n미만이면 거절", fontsize=9, color=INK2, va="top")
        ax.scatter(ans, [1] * len(ans), s=120, color=GRAY, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(10, 1.22, f"{len(ans)}개 모두 {min(ans)}점" if len(set(ans)) == 1 else f"{len(ans)}개", ha="center", fontsize=10, color=INK)
        seen = {}
        for d in sorted(oos, key=lambda d: (d["score"], d["id"])):
            k = seen.get(d["score"], 0); seen[d["score"]] = k + 1
            y = 0 - 0.17 * k  # 같은 점수는 아래로 쌓는다
            ax.scatter([d["score"]], [y], s=120, color=VIOLET, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(d["score"] + 0.28, y, d["id"], va="center", fontsize=9.5, color=INK)
        ax.set_yticks([1, -0.25])
        ax.set_yticklabels([f"정답 있는 질문\n({len(ans)}개)", f"범위 밖 질문\n({len(oos)}개)"], fontsize=10.5)
        ax.set_ylim(-0.9, 1.7)
        ax.set_xlabel("rerank 최고 점수 (0~10)")
        refused = sum(d["score"] < 5 for d in oos)
        ax.set_title(f"{LABEL[n].split(' (')[0]}: 기준 5에서 {refused}/{len(oos)}개 거절", loc="left", fontsize=12, fontweight="bold", pad=8)
    fig.suptitle("rerank 점수로 범위 밖 질문을 걸러낼 수 있는가", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.005, "점수 10인 범위 밖 질문(q37 영향평가 방법, q38 세부 기준 수치, q39 과징금 등 4개)은 정답 질문과 점수로 구분되지 않는다. 구성마다 10점이 되는 질문이 조금 다르다.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    return fig


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    final = "hybrid+rerank" if "hybrid+rerank" in data["configs"] else "hybrid"
    figs = {"eval_stages.png": fig_stages(data), "eval_by_type.png": fig_types(data, final)}
    gate = fig_gate(data)
    if gate is not None:
        figs["eval_gate.png"] = gate
    if all("ctx_select" in r for r in data["configs"][final]["rows"]):
        figs["eval_context_select.png"] = fig_ctx(data)
    for name, fig in figs.items():
        fig.savefig(OUT / name, dpi=150)
        print("saved", OUT / name)


if __name__ == "__main__":
    main()
