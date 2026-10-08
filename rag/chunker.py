"""Canonical Legal JSON → Parent-Child 청크.

- Parent = 조(條) 전체. LLM에 근거로 넘기는 단위.
- Child  = 검색(임베딩)하는 단위. 조가 짧으면 조 전체, 길면 항 → 호 → 목 순으로 필요한 만큼만 쪼갠다.
  쪼갤 때 상위 문장(항·호의 도입부)을 앞에 붙여 child 하나만 읽어도 뜻이 통하게 한다.
  문장 중간은 자르지 않으므로 단일 목이 max_chars 를 넘어도 그대로 둔다.
"""

import re
import uuid
from dataclasses import asdict, dataclass, field

CIRCLED_TO_INT = {c: i + 1 for i, c in enumerate(
    "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿"
)}
DEFAULT_MAX_CHARS = 300  # eval 비교(docs/chunking_experiments.md)에서 500·800·1200보다 안정적이었다
_NAMESPACE = uuid.UUID("6f1c1d0e-5b7a-4a43-9d57-2f6b8e0c3a11")


@dataclass
class Chunk:
    chunk_id: str  # 사람이 읽는 ID. 예: 인공지능기본법:제6조:②:4의2
    parent_id: str  # 조 단위 ID. 예: 인공지능기본법:제6조
    citation: str  # 인용 표기. 예: 인공지능기본법 제6조제2항제4호의2
    text: str  # child 원문 (도입부 포함)
    embed_text: str  # 임베딩·sparse 입력 (법령명 > 장 > 조 맥락 + text)
    parent_text: str  # 조 전체 원문
    metadata: dict = field(default_factory=dict)

    @property
    def point_id(self) -> str:
        """Qdrant point ID. chunk_id 에서 결정적으로 만들어 재실행해도 같은 포인트를 덮어쓴다."""
        return str(uuid.uuid5(_NAMESPACE, self.chunk_id))

    def payload(self) -> dict:
        d = asdict(self)
        meta = d.pop("metadata")
        return {**d, **meta}


# ----- 표기 / 렌더링 -------------------------------------------------------

def _para_ref(number: str) -> str:
    return f"제{CIRCLED_TO_INT[number]}항"


def _item_ref(number: str) -> str:
    if "의" in number:
        base, branch = number.split("의", 1)
        return f"제{base}호의{branch}"
    return f"제{number}호"


def _subitem_ref(number: str) -> str:
    return f"{number}목"


def _article_head(art: dict) -> str:
    return f"{art['label']}({art['title']})" if art["title"] != "삭제" else f"{art['label']} 삭제"


def _render_subitem(sub: dict, indent: int = 4) -> str:
    return f"{' ' * indent}{sub['number']}. {sub['text']}"


def _render_item(item: dict, indent: int = 2) -> str:
    lines = [f"{' ' * indent}{item['number']}. {item['text']}"]
    lines += [_render_subitem(s, indent + 2) for s in item["subitems"]]
    return "\n".join(lines)


def _render_paragraph(para: dict) -> str:
    lines = [f"{para['number']} {para['text']}".rstrip()]
    lines += [_render_item(i) for i in para["items"]]
    return "\n".join(lines)


def render_article(art: dict) -> str:
    """조 전체를 읽기 좋은 텍스트로 만든다 (parent_text)."""
    head = _article_head(art)
    if art["text"]:
        head = f"{head} {art['text']}"
    lines = [head]
    lines += [_render_item(i) for i in art["items"]]
    lines += [_render_paragraph(p) for p in art["paragraphs"]]
    return "\n".join(lines)


# ----- 청킹 ---------------------------------------------------------------

def chunk_law(
    doc: dict, max_chars: int = DEFAULT_MAX_CHARS, resolve_refs: bool = False
) -> list[Chunk]:
    law = doc["law"]
    name = law["short_name"] or law["title"]
    titles = {a["label"]: a["title"] for a in doc["articles"]}
    chunks: list[Chunk] = []
    for art in doc["articles"]:
        chunks.extend(_chunk_article(art, law, name, max_chars, titles if resolve_refs else None))
    for i, add in enumerate(doc.get("addenda", []), start=1):
        chunks.append(_addendum_chunk(add, i, law, name))
    _check_unique(chunks)
    return chunks


def _chunk_article(
    art: dict, law: dict, name: str, max_chars: int, titles: dict | None = None
) -> list[Chunk]:
    parent_id = f"{name}:{art['label']}"
    parent_text = render_article(art)
    head = _article_head(art)
    context = " > ".join(x for x in (name, art["chapter"], art["section"], head) if x)

    def make(path: list[str], refs: list[str], text: str) -> Chunk:
        # path: chunk_id 용 ["②", "4의2"], refs: 인용용 ["제2항", "제4호의2"]
        cid = ":".join([parent_id, *path])
        citation = f"{name} {art['label']}" + "".join(refs)
        meta = {
            "law_title": law["title"],
            "short_name": name,
            "law_no": law["law_no"],
            "effective_date": law["effective_date"],
            "chapter": art["chapter"],
            "section": art["section"],
            "article": art["label"],
            "article_title": art["title"],
            "article_number": art["number"],
            "paragraph": path[0] if path and path[0] in CIRCLED_TO_INT else None,
            "level": _level(path),
        }
        loc = " ".join(refs)
        embed = f"{context}{(' ' + loc) if loc else ''}\n{text}"
        if titles is not None:  # 참조 풀기: 임베딩 입력에만 덧붙이고 text·citation 은 그대로 둔다.
            body = text.replace(head, "", 1)  # 조 머리글("제7조(…)")은 참조가 아니다
            embed += "".join(f"\n{line}" for line in _ref_lines(body, art, titles))
        return Chunk(cid, parent_id, citation, text, embed, parent_text, meta)

    # 조 전체가 충분히 짧으면 child 하나
    if len(parent_text) <= max_chars:
        return [make([], [], parent_text)]

    out: list[Chunk] = []
    art_intro = f"{head} {art['text']}".strip()

    # 항 없이 호가 바로 붙는 조 (예: 제2조)
    if art["items"]:
        out += _chunk_items(art["items"], art_intro, [], [], make, max_chars)
    elif art["text"]:
        out.append(make([], [], f"{head} {art['text']}"))

    for para in art["paragraphs"]:
        p_path, p_refs = [para["number"]], [_para_ref(para["number"])]
        # 첫 항은 조 제목을 붙이고, 나머지 항도 child 단독 이해를 위해 조 제목을 붙인다.
        p_intro = f"{head} {para['number']} {para['text']}".strip()
        if len(_render_paragraph(para)) <= max_chars or not para["items"]:
            out.append(make(p_path, p_refs, f"{head}\n{_render_paragraph(para)}"))
        else:
            out += _chunk_items(para["items"], p_intro, p_path, p_refs, make, max_chars)
    return out


def _chunk_items(items, intro, path, refs, make, max_chars) -> list[Chunk]:
    out: list[Chunk] = []
    for item in items:
        i_path, i_refs = path + [item["number"]], refs + [_item_ref(item["number"])]
        i_intro = f"{intro}\n{_render_item(item, 2).splitlines()[0]}"
        full = f"{intro}\n{_render_item(item, 2)}"
        if len(full) <= max_chars or not item["subitems"]:
            out.append(make(i_path, i_refs, full))
            continue
        for sub in item["subitems"]:
            out.append(
                make(
                    i_path + [sub["number"]],
                    i_refs + [_subitem_ref(sub["number"])],
                    f"{i_intro}\n{_render_subitem(sub, 4)}",
                )
            )
    return out


_RE_XREF = re.compile(r"제(\d+)조(?:의(\d+))?")
_RE_LOCAL = re.compile(r"(?<![0-9조])제(\d+)항(?:제(\d+)호(?:의(\d+))?)?")
_INT_TO_CIRCLED = {v: k for k, v in CIRCLED_TO_INT.items()}
_REF_SNIPPET = 120
_MAX_REFS = 3


def _ref_lines(text: str, art: dict, titles: dict[str, str]) -> list[str]:
    """child 가 가리키는 조항의 뜻을 임베딩 입력용으로 풀어 쓴다.

    - 다른 조 참조("제31조제1항을 위반하여"): 그 조의 제목을 붙인다.
    - 같은 조 안의 항·호 참조("제4항제4호에 따른 위원"): 가리키는 항·호의 본문 앞부분을 붙인다.
      (한 child 에 이 법의 다른 조 참조가 섞여 있으면 항 번호가 어느 조의 것인지 모호해 생략한다.)
    다른 법령 참조(「…법」 제N조, 같은 법 제N조)는 이 법의 조문이 아니므로 제외한다.
    """
    lines: list[str] = []
    cross: dict[str, None] = {}
    for m in _RE_XREF.finditer(text):
        before = text[max(0, m.start() - 5) : m.start()]
        if before.rstrip().endswith("」") or before.endswith("같은 법 "):
            continue
        label = m.group(0)
        if label != art["label"] and label in titles:
            cross[label] = None
    lines += [f"[참조] {label}({titles[label]})" for label in cross][:_MAX_REFS]
    if cross:
        return lines

    paras = {p["number"]: p for p in art["paragraphs"]}
    seen: dict[str, None] = {}
    for m in _RE_LOCAL.finditer(text):
        para = paras.get(_INT_TO_CIRCLED.get(int(m.group(1)), ""))
        if para is None or m.group(0) in seen:
            continue
        seen[m.group(0)] = None
        target = para["text"]
        if m.group(2):
            num = m.group(2) + (f"의{m.group(3)}" if m.group(3) else "")
            item = next((i for i in para["items"] if i["number"] == num), None)
            if item is None:
                continue
            target = item["text"]
        if target[:30] in text:  # 가리키는 내용이 이미 이 child 안에 있으면 중복이다
            continue
        lines.append(f"[참조] {m.group(0)}: {target[:_REF_SNIPPET]}")
    return lines[:_MAX_REFS]


def _level(path: list[str]) -> str:
    # path 길이만으로는 "항 없는 조의 호"와 "항"을 구분할 수 없어 첫 요소로 판별한다.
    if not path:
        return "article"
    has_para = path[0] in CIRCLED_TO_INT
    depth = len(path) - (1 if has_para else 0)
    if has_para:
        return ["paragraph", "item", "subitem"][min(depth, 2)]
    return ["item", "subitem"][min(depth - 1, 1)]


def _addendum_chunk(add: dict, index: int, law: dict, name: str) -> Chunk:
    text = f"{add['title']}\n{add['text']}".strip()
    parent_id = f"{name}:부칙{index}"
    meta = {
        "law_title": law["title"], "short_name": name, "law_no": law["law_no"],
        "effective_date": law["effective_date"], "chapter": None, "section": None,
        "article": "부칙", "article_title": "부칙", "article_number": None,
        "paragraph": None, "level": "addendum",
    }
    return Chunk(parent_id, parent_id, f"{name} 부칙", text, f"{name} > 부칙\n{text}", text, meta)


def _check_unique(chunks: list[Chunk]) -> None:
    seen: set[str] = set()
    for c in chunks:
        if c.chunk_id in seen:
            raise ValueError(f"chunk_id 가 중복됐다: {c.chunk_id}")
        seen.add(c.chunk_id)
