"""LlamaIndex 문서 로딩 - 한국어 RAG 특화."""

import csv
import json
import re
from pathlib import Path

from llama_index.core.schema import TextNode

import config
from schema import RAGDocument, expand_keywords


def convert_date_to_korean(text: str) -> str:
    """Convert YYYY-MM-DD dates to Korean format (YYYY년 M월 D일)."""

    def replace_date(match):
        year, month, day = match.groups()
        return f"{year}년 {int(month)}월 {int(day)}일"

    return re.sub(r"(\d{4})-(\d{2})-(\d{2})", replace_date, text)


def extract_date_metadata(date_str: str) -> dict:
    """Extract year, month, day, date_int, and weekday info from YYYY-MM-DD date string."""
    from datetime import datetime as dt

    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if match:
        year, month, day = match.groups()
        # YYYYMMDD 정수형 (정밀한 날짜 비교용)
        date_int = int(f"{year}{month}{day}")

        # 요일 계산 (0=월요일, 6=일요일)
        date_obj = dt(int(year), int(month), int(day))
        day_of_week = date_obj.weekday()

        return {
            "year": int(year),
            "month": int(month),
            "day": int(day),
            "start_date": date_str,  # 원본 문자열
            "start_date_int": date_int,  # 정수형 (20250215 형식)
            "day_of_week": day_of_week,  # 0=월, 1=화, ..., 6=일
            "is_weekend": day_of_week >= 5,  # 토(5), 일(6) = True
        }
    return {}


def extract_registration_metadata(row: dict) -> dict:
    """등록 기간 메타데이터 추출 (정수형 날짜로 비교 가능하게)."""
    reg_start = row.get("등록 시작일", "")
    reg_end = row.get("등록 마감일", "")

    result = {}

    # 등록 시작일 정수화
    if reg_start:
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})", reg_start)
        if match:
            result["reg_start_int"] = int(f"{match.group(1)}{match.group(2)}{match.group(3)}")

    # 등록 마감일 정수화
    if reg_end:
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})", reg_end)
        if match:
            result["reg_end_int"] = int(f"{match.group(1)}{match.group(2)}{match.group(3)}")

    return result


def extract_duration_metadata(row: dict) -> dict:
    """행사 기간(일 수) 계산."""
    from datetime import datetime as dt

    start = row.get("행사 시작일", "")
    end = row.get("행사 종료일", "")

    if start and end:
        try:
            start_dt = dt.strptime(start, "%Y-%m-%d")
            end_dt = dt.strptime(end, "%Y-%m-%d")
            duration = (end_dt - start_dt).days + 1
            return {"duration_days": max(1, duration)}
        except ValueError:
            pass

    return {"duration_days": 1}  # 기본값: 하루


def normalize_location(location: str) -> str:
    """Normalize location string for consistent matching."""
    if not location:
        return ""

    # 1. 연속 공백을 단일 공백으로
    normalized = re.sub(r"\s+", " ", location.strip())

    # 2. "aT 센터" → "aT센터" (공백 제거)
    normalized = re.sub(r"aT\s*센터", "aT센터", normalized)

    # 3. "창조룸Ⅰ" → "창조룸 Ⅰ" (로마숫자 앞 공백 통일)
    normalized = re.sub(r"창조룸\s*([ⅠⅡⅢⅣⅤⅰⅱⅲⅳⅴ])", r"창조룸 \1", normalized)

    # 4. "세계로룸Ⅰ" → "세계로룸 Ⅰ"
    normalized = re.sub(r"세계로룸\s*([ⅠⅡⅢⅣⅤⅰⅱⅲⅳⅴ])", r"세계로룸 \1", normalized)

    return normalized


def extract_category_from_event(event_name: str) -> str:
    """Extract event category from event name."""
    event_name_lower = event_name.lower()

    category_patterns = [
        (r"심포지엄|symposium", "심포지엄"),
        (r"워크숍|workshop", "워크숍"),
        (r"스쿨|school", "스쿨"),
        (r"학술대회|conference", "학술대회"),
        (r"교육|training|리더쉽", "교육"),
        (r"세미나|seminar", "세미나"),
    ]

    for pattern, category in category_patterns:
        if re.search(pattern, event_name_lower):
            return category

    return "기타"


def extract_keywords_from_event(row: dict) -> list[str]:
    """Extract keywords from event data."""
    keywords = []

    event_name = row.get("행사명", "")
    if event_name:
        medical_patterns = [
            r"(천식|COPD|ILD|NTM|폐암|결핵|폐기능|수면|호흡|금연|기침|폐혈관)",
            r"(기관지확장증|감염병|환경성폐질환|분자폐암)",
        ]
        for pattern in medical_patterns:
            matches = re.findall(pattern, event_name, re.IGNORECASE)
            keywords.extend(matches)

        event_type_matches = re.findall(
            r"(심포지엄|워크숍|학술대회|교육|스쿨|세미나)", event_name
        )
        keywords.extend(event_type_matches)

        org_matches = re.findall(r"(연구회|학회)", event_name)
        keywords.extend(org_matches)

    location = row.get("행사장소", "")
    if location:
        location_keywords = re.findall(r"(양재|aT센터|서울대|중앙대|성모병원|SC)", location)
        keywords.extend(location_keywords)

    keywords = list(set(keywords))
    return expand_keywords(keywords)


def generate_question_from_event(row: dict) -> str:
    """Generate a natural question from event data."""
    event_name = row.get("행사명", "")
    start_date = row.get("행사 시작일", "")

    if start_date:
        date_kr = convert_date_to_korean(start_date)
        match = re.match(r"(\d{4})년 (\d{1,2})월", date_kr)
        if match:
            year, month = match.groups()
            return f"{year}년 {month}월 {event_name} 일정과 장소는?"

    return f"{event_name} 일정과 장소는?"


def generate_answer_from_event(row: dict) -> str:
    """Generate a structured answer from event data."""
    event_name = row.get("행사명", "")
    start_date = convert_date_to_korean(row.get("행사 시작일", ""))
    end_date = convert_date_to_korean(row.get("행사 종료일", ""))
    location = row.get("행사장소", "")
    reg_start = convert_date_to_korean(row.get("등록 시작일", ""))
    reg_end = convert_date_to_korean(row.get("등록 마감일", ""))

    answer_parts = [f"{event_name}"]

    if start_date:
        if start_date == end_date or not end_date:
            answer_parts.append(f"일시: {start_date}")
        else:
            answer_parts.append(f"일시: {start_date} ~ {end_date}")

    if location:
        answer_parts.append(f"장소: {location}")

    if reg_start and reg_end:
        answer_parts.append(f"등록기간: {reg_start} ~ {reg_end}")

    return "\n".join(answer_parts)


def generate_explanation_from_event(row: dict) -> str:
    """Generate explanation from event data."""
    credits = row.get("평점", "")
    url = row.get("url", "")

    explanation_parts = []
    if credits:
        explanation_parts.append(f"평점: {credits}")
    if url:
        explanation_parts.append(f"URL: {url}")

    return "\n".join(explanation_parts) if explanation_parts else ""


def build_node_text(row: dict, keywords: list[str]) -> str:
    """Build Key-Value format text content for embedding."""
    parts = []

    # 행사명
    event_name = row.get("행사명", "")
    if event_name:
        parts.append(f"행사명: {event_name}")

    # 날짜 정보
    start_date = row.get("행사 시작일", "")
    end_date = row.get("행사 종료일", "")
    if start_date:
        parts.append(f"행사 시작일: {start_date}")
    if end_date and end_date != start_date:
        parts.append(f"행사 종료일: {end_date}")

    # 장소
    location = row.get("행사장소", "")
    if location:
        parts.append(f"행사장소: {location}")

    # 등록 기간
    reg_start = row.get("등록 시작일", "")
    reg_end = row.get("등록 마감일", "")
    if reg_start:
        parts.append(f"등록 시작일: {reg_start}")
    if reg_end:
        parts.append(f"등록 마감일: {reg_end}")

    # 평점
    credits = row.get("평점", "")
    if credits:
        parts.append(f"평점: {credits}")

    # URL
    url = row.get("url", "")
    if url:
        parts.append(f"URL: {url}")

    # 카테고리
    category = extract_category_from_event(event_name)
    parts.append(f"카테고리: {category}")

    # 키워드
    if keywords:
        parts.append(f"키워드: {', '.join(keywords)}")

    return "\n".join(parts)


def load_csv(file_path: str) -> list[TextNode]:
    """Load CSV file and create TextNodes with metadata."""
    nodes = []
    path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            # 데이터 추출
            event_name = row.get("행사명", "")
            start_date = row.get("행사 시작일", "")
            date_meta = extract_date_metadata(start_date)
            reg_meta = extract_registration_metadata(row)  # [신규] 등록 기간
            duration_meta = extract_duration_metadata(row)  # [신규] 행사 기간
            category = extract_category_from_event(event_name)
            keywords = extract_keywords_from_event(row)

            # Key-Value 형식 노드 텍스트 생성
            text = build_node_text(row, keywords)

            # 답변 템플릿 (LLM 컨텍스트용)
            answer = generate_answer_from_event(row)

            # 메타데이터 구성
            metadata = {
                "source": str(path),
                "filename": path.name,
                "row": idx,
                "event_name": event_name,
                "category": category,
                "location": normalize_location(row.get("행사장소", "")),
                "year": date_meta.get("year"),
                "month": date_meta.get("month"),
                "day": date_meta.get("day"),
                "start_date": date_meta.get("start_date"),  # YYYY-MM-DD 문자열
                "start_date_int": date_meta.get("start_date_int"),  # YYYYMMDD 정수
                "day_of_week": date_meta.get("day_of_week"),  # 0=월 ~ 6=일
                "is_weekend": date_meta.get("is_weekend"),  # 주말 여부
                "reg_start_int": reg_meta.get("reg_start_int"),  # 등록 시작일 (정수)
                "reg_end_int": reg_meta.get("reg_end_int"),  # 등록 마감일 (정수)
                "duration_days": duration_meta.get("duration_days"),  # 행사 기간 (일)
                "credits": row.get("평점", ""),
                "url": row.get("url", ""),
                "keywords": ",".join(keywords),
                "answer_template": answer,
            }

            # None 값 제거
            metadata = {k: v for k, v in metadata.items() if v is not None}

            # TextNode 생성
            node = TextNode(
                text=text,
                id_=f"csv_{path.stem}_{idx}",
                metadata=metadata,
            )
            nodes.append(node)

    return nodes


def load_jsonl(file_path: str) -> list[TextNode]:
    """Load JSONL file and create TextNodes."""
    nodes = []
    path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                rag_doc = RAGDocument(**data)

                # Key-Value 형식 텍스트 생성
                parts = []
                if rag_doc.metadata.event_name:
                    parts.append(f"행사명: {rag_doc.metadata.event_name}")
                if rag_doc.metadata.start_date:
                    parts.append(f"행사 시작일: {rag_doc.metadata.start_date}")
                if rag_doc.metadata.end_date:
                    parts.append(f"행사 종료일: {rag_doc.metadata.end_date}")
                if rag_doc.metadata.location:
                    parts.append(f"행사장소: {rag_doc.metadata.location}")
                if rag_doc.metadata.credits:
                    parts.append(f"평점: {rag_doc.metadata.credits}")
                if rag_doc.metadata.url:
                    parts.append(f"URL: {rag_doc.metadata.url}")
                if rag_doc.metadata.category:
                    parts.append(f"카테고리: {rag_doc.metadata.category}")
                if rag_doc.keywords:
                    parts.append(f"키워드: {', '.join(rag_doc.keywords)}")
                text = "\n".join(parts)

                # start_date_int 계산
                start_date_int = None
                start_date = None
                if rag_doc.search_boost.year and rag_doc.search_boost.month and rag_doc.search_boost.day:
                    start_date = f"{rag_doc.search_boost.year:04d}-{rag_doc.search_boost.month:02d}-{rag_doc.search_boost.day:02d}"
                    start_date_int = int(f"{rag_doc.search_boost.year:04d}{rag_doc.search_boost.month:02d}{rag_doc.search_boost.day:02d}")

                metadata = {
                    "source": str(path),
                    "filename": path.name,
                    "row": line_num,
                    "event_name": rag_doc.metadata.event_name or "",
                    "category": extract_category_from_event(rag_doc.metadata.event_name or ""),
                    "location": rag_doc.search_boost.location_normalized or "",
                    "year": rag_doc.search_boost.year,
                    "month": rag_doc.search_boost.month,
                    "day": rag_doc.search_boost.day,
                    "start_date": start_date,
                    "start_date_int": start_date_int,
                    "keywords": ",".join(rag_doc.keywords),
                    "answer_template": rag_doc.content.answer,
                }

                metadata = {k: v for k, v in metadata.items() if v is not None}

                node = TextNode(
                    text=text,
                    id_=rag_doc.id,
                    metadata=metadata,
                )
                nodes.append(node)

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing line {line_num} in {file_path}: {e}")

    return nodes


def load_markdown(file_path: str) -> list[TextNode]:
    """Load markdown file and create TextNode."""
    path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    node = TextNode(
        text=content,
        id_=f"md_{path.stem}",
        metadata={
            "source": str(path),
            "filename": path.name,
        },
    )
    return [node]


def load_documents_from_dir(docs_dir: str | None = None) -> list[TextNode]:
    """Load all documents from the docs directory."""
    if docs_dir is None:
        docs_dir = config.DOCS_DIR

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return []

    nodes = []
    for file_path in docs_path.rglob("*"):
        try:
            if file_path.suffix == ".csv":
                nodes.extend(load_csv(str(file_path)))
            elif file_path.suffix == ".jsonl":
                nodes.extend(load_jsonl(str(file_path)))
            elif file_path.suffix == ".md":
                nodes.extend(load_markdown(str(file_path)))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return nodes
