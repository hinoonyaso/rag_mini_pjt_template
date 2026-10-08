"""국가법령정보 Open API(XML) → Canonical Legal JSON.

lawService.do?target=law&type=XML 응답은 조·항·호·목이 이미 태그로 나뉘어 있다.
  법령 / 기본정보, 조문 / 조문단위(조문여부: 전문=장·절 제목, 조문=조) / 항 / 호 / 목, 부칙 / 부칙단위
이 모듈은 그 구조를 아래 Canonical JSON 으로 옮기고, 번호 접두어·개정 이력 표기를 본문에서 분리한다.

Canonical JSON 구조:
    {
      "law": {title, short_name, law_no, law_id, mst, effective_date, amended_date, source},
      "articles": [
        {id, number, branch, label, title, chapter, section, text, notes,
         items: [호], paragraphs: [{number, text, notes, items: [호]}]}
      ],
      "addenda": [{title, text}]
    }
    호(item) = {number, text, notes, subitems: [{number, text, notes}]}  (subitem = 목)
    번호 없는 항(예: 제2조)의 호는 article.items 에 들어간다.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from common.config import (
    LAW_API_OC,
    LAW_API_URL,
    LAW_EFFECTIVE_DATE,
    LAW_MST,
    LAW_RAW_XML_PATH,
)

# 개정 이력 표기: <개정 2026.1.20>, <신설 ...>, [본조신설 ...], [제목개정 ...]
RE_NOTE = re.compile(
    r"<(?:전문개정|개정|신설|삭제|본조신설)[^>]*>|\[(?:제목개정|본조신설|본조변경|전문개정)[^\]]*\]"
)
RE_ARTICLE_HEAD = re.compile(r"^제\d+조(?:의\d+)?(?:\([^)]*\))?\s*")
RE_HEADING = re.compile(r"^제(\d+)(장|절|관)\s+(.+)$")
RE_LABEL_ITEM = re.compile(r"^\d+(?:의\d+)*\.\s*")  # "4." "4의2."
RE_LABEL_SUBITEM = re.compile(r"^[가-힣](?:의\d+)?\.\s*")  # "가." "가의2."


# ----- API 호출 -------------------------------------------------------------

def fetch_law_xml(
    mst: str = LAW_MST,
    ef_yd: str = LAW_EFFECTIVE_DATE,
    oc: str | None = LAW_API_OC,
    timeout: int = 30,
) -> bytes:
    """법령 본문 XML 을 내려받는다. OC 가 URL 에 들어가므로 예외 메시지에 URL 을 싣지 않는다."""
    if not oc:
        raise RuntimeError("LAW_API_OC 가 없다. .env 에 LAW_API_OC=<발급받은 OC> 를 넣어라.")
    query = urllib.parse.urlencode(
        {"OC": oc, "target": "law", "type": "XML", "MST": mst, "efYd": ef_yd}
    )
    try:
        with urllib.request.urlopen(f"{LAW_API_URL}?{query}", timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"법령 API HTTP 오류: {e.code}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"법령 API 연결 실패: {e.reason}") from None

    # 인증·IP 등록 오류는 HTTP 200 으로 오류 문서(HTML/XML)를 돌려주므로 루트 태그로 판별한다.
    try:
        root_tag = ET.fromstring(raw).tag
    except ET.ParseError:
        root_tag = None
    if root_tag != "법령":
        head = raw[:200].decode("utf-8", "replace").replace(oc, "***")
        raise RuntimeError(f"법령 본문 XML 이 아니다 (OC 인증/IP 등록/MST 확인): {head!r}")
    return raw


# ----- 파싱 -----------------------------------------------------------------

def _text(el: ET.Element | None, tag: str) -> str:
    child = el.find(tag) if el is not None else None
    return (child.text or "").strip() if child is not None else ""


def _clean(text: str) -> tuple[str, list[str]]:
    """개정 이력 표기를 본문에서 분리하고 공백을 정리한다."""
    notes = [re.sub(r"\s+", " ", n) for n in RE_NOTE.findall(text)]
    text = RE_NOTE.sub("", text)
    text = re.sub(r"[ \t 　]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text, notes


def _iso_date(yyyymmdd: str) -> str | None:
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", yyyymmdd or "")
    return "-".join(m.groups()) if m else None


def _amend_note(el: ET.Element, kind_tag: str, date_tag: str) -> list[str]:
    kind, date = _text(el, kind_tag), _text(el, date_tag)
    return [f"{kind} {date}".strip()] if kind else []


def _branch_number(el: ET.Element, number_tag: str, branch_tag: str) -> str:
    """'4.' + 가지번호 2 → '4의2' 처럼 번호만 남긴다."""
    number = _text(el, number_tag).rstrip(".").strip()
    branch = _text(el, branch_tag)
    return f"{number}의{branch}" if branch else number


def _parse_items(parent: ET.Element) -> list[dict]:
    items = []
    for ho in parent.findall("호"):
        text, notes = _clean(RE_LABEL_ITEM.sub("", _text(ho, "호내용"), count=1))
        subitems = []
        for mok in ho.findall("목"):
            s_text, s_notes = _clean(RE_LABEL_SUBITEM.sub("", _text(mok, "목내용"), count=1))
            subitems.append(
                {"number": _branch_number(mok, "목번호", "목가지번호"), "text": s_text, "notes": s_notes}
            )
        items.append(
            {"number": _branch_number(ho, "호번호", "호가지번호"), "text": text, "notes": notes, "subitems": subitems}
        )
    return items


def parse_law_xml(raw: bytes | str, source: str = "", effective_date: str | None = None) -> dict:
    """법령 본문 XML 을 Canonical Legal JSON(dict)으로 변환한다."""
    root = ET.fromstring(raw)
    info = root.find("기본정보")
    if info is None or root.find("조문") is None:
        raise ValueError("기본정보 또는 조문이 없는 XML 이다.")

    law = {
        "title": _text(info, "법령명_한글"),
        "short_name": _text(info, "법령명약칭") or None,
        "law_no": f"{_text(info, '법종구분')} 제{_text(info, '공포번호')}호",
        "law_id": _text(info, "법령ID"),
        "mst": LAW_MST,
        "effective_date": _iso_date(effective_date or "") or _iso_date(_text(info, "시행일자")),
        "amended_date": _iso_date(_text(info, "공포일자")),
        "source": source,
    }
    if not law["title"]:
        raise ValueError("법령명을 찾지 못했다.")

    articles: list[dict] = []
    chapter = section = None
    for unit in root.find("조문").findall("조문단위"):
        kind = _text(unit, "조문여부")
        content = _text(unit, "조문내용")

        if kind == "전문":  # 장·절 제목
            m = RE_HEADING.match(content)
            if m:
                label = f"제{m.group(1)}{m.group(2)} {m.group(3).strip()}"
                if m.group(2) == "장":
                    chapter, section = label, None
                else:
                    section = label
            continue
        if kind != "조문":
            continue

        number = int(_text(unit, "조문번호"))
        branch = _text(unit, "조문가지번호")
        label = f"제{number}조" + (f"의{branch}" if branch else "")
        body, notes = _clean(RE_ARTICLE_HEAD.sub("", content, count=1))
        title = _text(unit, "조문제목") or "삭제"
        notes += _amend_note(unit, "조문제개정유형", "조문제개정일자문자열")
        notes += [n for ref in unit.findall("조문참고자료") for n in _clean(ref.text or "")[1]]

        art = {
            "id": label, "number": number, "branch": int(branch) if branch else None,
            "label": label, "title": title, "chapter": chapter, "section": section,
            "text": body, "notes": notes, "items": [], "paragraphs": [],
        }
        for hang in unit.findall("항"):
            h_number = _text(hang, "항번호")
            h_text, h_notes = _clean(_text(hang, "항내용"))
            h_notes += _amend_note(hang, "항제개정유형", "항제개정일자문자열")
            if h_number:
                if h_text.startswith(h_number):
                    h_text = h_text[len(h_number):].lstrip()
                art["paragraphs"].append(
                    {"number": h_number, "text": h_text, "notes": h_notes, "items": _parse_items(hang)}
                )
            else:  # 번호 없는 항: 호는 조 직속, 본문이 있으면 조 본문에 이어 붙인다.
                if h_text:
                    art["text"] = f"{art['text']}\n{h_text}".strip()
                art["notes"] += h_notes
                art["items"] += _parse_items(hang)
        articles.append(art)

    addenda = []
    for unit in (root.find("부칙").findall("부칙단위") if root.find("부칙") is not None else []):
        content = _text(unit, "부칙내용")
        title, _, rest = content.partition("\n")
        addenda.append({"title": _clean(title)[0], "text": _clean(rest)[0]})

    doc = {"law": law, "articles": articles, "addenda": addenda}
    _validate(doc)
    return doc


def _validate(doc: dict) -> None:
    """조 번호 순서와 빈 조문을 확인한다. 어긋나면 파싱 결과를 신뢰할 수 없으므로 실패시킨다."""
    arts = doc["articles"]
    if not arts:
        raise ValueError("조문을 하나도 찾지 못했다.")
    prev = (0, 0)
    for a in arts:
        key = (a["number"], a["branch"] or 0)
        if key <= prev:
            raise ValueError(f"조 번호 순서가 어긋났다: {a['label']} (이전 {prev})")
        prev = key
        if not (a["text"] or a["paragraphs"] or a["items"]):
            raise ValueError(f"본문이 비어 있다: {a['label']}")


# ----- 로드 / 저장 -----------------------------------------------------------

def load_law(refresh: bool = False, raw_path: str | Path = LAW_RAW_XML_PATH) -> dict:
    """원본 XML 캐시가 있으면 그것을 파싱하고, 없거나 refresh=True 면 API 에서 새로 받는다."""
    raw_path = Path(raw_path)
    if refresh or not raw_path.exists():
        raw = fetch_law_xml()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)
    else:
        raw = raw_path.read_bytes()
    return parse_law_xml(raw, source=f"law.go.kr MST={LAW_MST} efYd={LAW_EFFECTIVE_DATE}",
                         effective_date=LAW_EFFECTIVE_DATE)


def save_canonical(doc: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_canonical(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
