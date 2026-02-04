"""LlamaIndex RAG 체인 - 메타데이터 필터링 지원."""

import re
from datetime import datetime, timedelta
from typing import Generator

from llama_index.core import PromptTemplate, Settings
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.llms.ollama import Ollama

import config
from vector_store import get_index, get_all_by_filter
from embeddings import get_embed_model


def get_today_korean() -> str:
    """Get today's date in Korean format."""
    today = datetime.now()
    return f"{today.year}년 {today.month}월 {today.day}일"

# 싱글톤 인스턴스
_llm: Ollama | None = None

def get_system_prompt() -> str:
    """Get system prompt with current date."""
    today = get_today_korean()
    today_date = datetime.now().strftime("%Y-%m-%d")
    return f"""당신은 사내 문서를 기반으로 질문에 답변하는 한국어 AI 어시스턴트입니다.

## 기준 정보
- 오늘 날짜: {today} ({today_date})
- 이 날짜를 기준으로 과거/미래, 등록 가능 여부 등을 판단하세요.

## 필수 규칙
1. 모든 답변은 반드시 한국어로만 작성하세요.
2. 검색된 문서에 없는 내용은 절대 지어내지 마세요.
3. 정보가 없으면 "해당 정보를 찾을 수 없습니다"라고 답변하세요.
4. 행사명, 고유명사, 장소명 등은 원문 그대로 유지하세요.

## 등록 상태 판단 기준 (오늘: {today_date})
- "등록 가능": 오늘이 등록시작일과 등록마감일 사이
- "마감 임박": 등록마감일이 7일 이내
- "등록 전": 등록시작일이 오늘 이후
- "마감됨": 등록마감일이 오늘 이전

## 답변 형식
- 여러 항목: 번호 매겨서 빠짐없이 전부 나열
- 표 요청 시: Markdown 표 형식 (| 컬럼1 | 컬럼2 |)
- URL 있으면: 함께 제공
- 등록기간 있으면: 함께 안내

## 중요: 필터링된 결과 처리
- 제공된 문서는 이미 질문 조건에 맞게 필터링된 결과입니다.
- 사용자가 "등록 가능한" 등 등록 상태를 명시하지 않았다면, 등록 마감 여부와 관계없이 모든 문서를 답변에 포함하세요.
- 등록상태는 참고 정보일 뿐, 답변에서 제외하는 기준이 아닙니다.

## 모호한 질문 처리
- "그거", "거기" 등 대상이 불명확하면 되물어보세요.
- 예: "어떤 행사를 말씀하시는 건가요?"
"""

def get_qa_prompt() -> PromptTemplate:
    """Get QA prompt with current date."""
    today = get_today_korean()
    return PromptTemplate(
        f"""\
오늘 날짜: {today}

다음은 질문과 관련된 문서 내용입니다:

{{context_str}}

위 문서 내용을 바탕으로 다음 질문에 답변하세요.
- 여러 항목이 있으면 전부 나열하세요.
- 반드시 한국어로만 답변하세요.
- "오늘", "가장 빠른" 등은 오늘 날짜({today}) 기준입니다.

질문: {{query_str}}

답변:"""
    )


def get_llm() -> Ollama:
    """Get the Ollama LLM instance (singleton)."""
    global _llm

    if _llm is None:
        print("[초기화] LLM 연결 중...")
        _llm = Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            request_timeout=120.0,
        )
        print("[초기화] LLM 연결 완료")

    return _llm


def setup_settings() -> None:
    """Configure global LlamaIndex settings."""
    Settings.llm = get_llm()
    Settings.embed_model = get_embed_model()
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP


def parse_date_from_query(query: str) -> tuple[int | None, int | None, list[int] | None]:
    """Extract year, month, and month range from query string.

    Returns:
        (year, month, month_range) - month_range is a list of months for range queries
    """
    year = None
    month = None
    month_range = None

    # 연도 파싱
    year_match = re.search(r"(\d{4})년", query)
    if year_match:
        year = int(year_match.group(1))

    # 기간 키워드 파싱 (상반기, 하반기, 분기 등)
    if re.search(r"상반기|1반기|전반기", query):
        month_range = [1, 2, 3, 4, 5, 6]
    elif re.search(r"하반기|2반기|후반기", query):
        month_range = [7, 8, 9, 10, 11, 12]
    elif re.search(r"1분기|1사분기", query):
        month_range = [1, 2, 3]
    elif re.search(r"2분기|2사분기", query):
        month_range = [4, 5, 6]
    elif re.search(r"3분기|3사분기", query):
        month_range = [7, 8, 9]
    elif re.search(r"4분기|4사분기", query):
        month_range = [10, 11, 12]

    # 명시적 월 범위 파싱 (예: "1월~6월", "3월부터 5월까지", "1월-6월")
    range_match = re.search(r"(\d{1,2})월\s*[~\-부터]\s*(\d{1,2})월", query)
    if range_match:
        start_month = int(range_match.group(1))
        end_month = int(range_match.group(2))
        if 1 <= start_month <= 12 and 1 <= end_month <= 12 and start_month <= end_month:
            month_range = list(range(start_month, end_month + 1))

    # 단일 월 파싱 (범위가 없는 경우에만)
    if month_range is None:
        month_match = re.search(r"(\d{1,2})월", query)
        if month_match:
            month = int(month_match.group(1))

    return year, month, month_range


def parse_category_from_query(query: str) -> str | None:
    """Extract event category from query string."""
    category_patterns = [
        (r"심포지엄|심포지움", "심포지엄"),
        (r"워크숍|워크샵", "워크숍"),
        (r"스쿨|school", "스쿨"),
        (r"학술대회", "학술대회"),
        (r"교육|연수|리더쉽", "교육"),
        (r"세미나", "세미나"),
    ]

    query_lower = query.lower()
    for pattern, category in category_patterns:
        if re.search(pattern, query_lower):
            return category

    return None


def parse_location_from_query(query: str) -> str | None:
    """Extract location keyword from query string."""
    location_patterns = [
        (r"양재\s*aT\s*센터", "양재 aT센터"),
        (r"서울대", "서울대"),
        (r"코엑스", "코엑스"),
        (r"벡스코", "벡스코"),
        (r"SC\s*컨벤션", "SC 컨벤션센터"),
        (r"성모병원", "성모병원"),
        (r"중앙대", "중앙대"),
    ]

    for pattern, normalized in location_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return normalized

    return None


def parse_weekend_filter(query: str) -> bool | None:
    """Extract weekend/weekday filter from query string.

    Returns:
        True: 주말만
        False: 평일만
        None: 필터 없음
    """
    if re.search(r"주말|토요일|일요일|토,?\s*일|토·일", query):
        return True
    if re.search(r"평일|월요일|화요일|수요일|목요일|금요일|월~금", query):
        return False
    return None


def parse_registration_filter(query: str) -> str | None:
    """등록 상태 키워드 파싱.

    Returns:
        "available": 현재 등록 가능
        "closing_soon": 마감 임박 (7일 이내)
        "upcoming": 등록 시작 전
        "exclude_closed": 마감된 것 제외
        None: 필터 없음
    """
    # "등록"이 포함된 경우에만 등록 상태 필터 적용
    if re.search(r"등록.*(가능|신청|접수)|지금.*(신청|등록)|당장.*(신청|등록)", query):
        return "available"
    if re.search(r"등록.*(마감|임박)|마감.*(임박|급|곧)|일주일.*(안|내).*마감", query):
        return "closing_soon"
    if re.search(r"등록.*(전|대기|시작.*전)|아직.*등록.*(안|전)", query):
        return "upcoming"
    if re.search(r"등록.*(마감|끝|지난).*(제외|빼)|마감.*제외", query):
        return "exclude_closed"
    return None


def parse_duration_filter(query: str) -> str | None:
    """행사 기간 필터 파싱.

    Returns:
        "multi_day": 며칠간 진행
        "single_day": 하루 행사
        None: 필터 없음
    """
    if re.search(r"며칠|여러\s*날|장기|이틀|[23]일|연속|동안\s*진행", query):
        return "multi_day"
    if re.search(r"하루|당일|단기|하루\s*만", query):
        return "single_day"
    return None


def parse_exclusion_filter(query: str) -> str | None:
    """제외 조건 파싱 (카테고리).

    Returns:
        제외할 카테고리명 또는 None
    """
    patterns = [
        (r"심포지엄.*(말고|제외|빼고|아니고|외)", "심포지엄"),
        (r"워크숍.*(말고|제외|빼고|아니고|외)", "워크숍"),
        (r"스쿨.*(말고|제외|빼고|아니고|외)", "스쿨"),
        (r"세미나.*(말고|제외|빼고|아니고|외)", "세미나"),
        (r"교육.*(말고|제외|빼고|아니고|외)", "교육"),
    ]
    for pattern, category in patterns:
        if re.search(pattern, query):
            return category
    return None


def build_metadata_filters(query: str) -> MetadataFilters | None:
    """Build metadata filters from query string (for LlamaIndex)."""
    filters = []

    # 연도 필터
    year, month, month_range = parse_date_from_query(query)
    if year:
        filters.append(MetadataFilter(key="year", value=year, operator=FilterOperator.EQ))
    if month:
        filters.append(MetadataFilter(key="month", value=month, operator=FilterOperator.EQ))
    # 참고: LlamaIndex MetadataFilter는 $in 미지원, month_range는 ChromaDB 직접 필터에서만 사용

    # 카테고리 필터
    category = parse_category_from_query(query)
    if category:
        filters.append(MetadataFilter(key="category", value=category, operator=FilterOperator.EQ))

    if filters:
        return MetadataFilters(filters=filters)
    return None


def is_time_based_query(query: str) -> bool:
    """Check if query is asking about upcoming/nearest events."""
    time_patterns = [
        r"가장\s*빠른",
        r"가장\s*빨리",
        r"가장\s*가까운",
        r"오늘\s*이후",
        r"내일\s*이후",
        r"다음\s*행사",
        r"가까운\s*행사",
        r"다가오는",
        r"예정된",
        r"곧\s*있는",
        r"앞으로",
        r"오늘\s*기준",
        r"이번\s*달",
        r"이번\s*주",
    ]
    for pattern in time_patterns:
        if re.search(pattern, query):
            return True
    return False


def build_chroma_filters(query: str) -> dict | None:
    """Build ChromaDB where clause from query string."""
    conditions = []

    # 연도/월/월범위 파싱
    year, month, month_range = parse_date_from_query(query)

    # 시간 기반 쿼리 처리 (오늘 이후 행사만)
    if is_time_based_query(query):
        today = datetime.now()
        today_int = int(today.strftime("%Y%m%d"))

        # 사용자가 명시한 연도/월이 과거인지 확인
        is_past_date = False
        if year and month:
            query_date_int = int(f"{year}{month:02d}28")
            is_past_date = query_date_int < today_int
        elif year and month_range:
            # 월 범위의 마지막 월 기준
            last_month = max(month_range)
            query_date_int = int(f"{year}{last_month:02d}28")
            is_past_date = query_date_int < today_int
        elif year:
            is_past_date = year < today.year

        # 과거 날짜가 아닌 경우에만 "오늘 이후" 필터 적용
        if not is_past_date:
            conditions.append({"start_date_int": {"$gte": today_int}})

    # 연도 필터
    if year:
        conditions.append({"year": {"$eq": year}})

    # 월 필터 (단일 또는 범위)
    if month_range:
        conditions.append({"month": {"$in": month_range}})
    elif month:
        conditions.append({"month": {"$eq": month}})

    # 주말/평일 필터
    is_weekend = parse_weekend_filter(query)
    if is_weekend is not None:
        conditions.append({"is_weekend": {"$eq": is_weekend}})

    # 카테고리 필터
    category = parse_category_from_query(query)
    if category:
        conditions.append({"category": {"$eq": category}})

    # 제외 조건 ($ne)
    exclusion = parse_exclusion_filter(query)
    if exclusion:
        conditions.append({"category": {"$ne": exclusion}})

    # 등록 상태 필터
    today_int = int(datetime.now().strftime("%Y%m%d"))
    reg_status = parse_registration_filter(query)
    if reg_status == "available":
        # 오늘이 등록 기간 내 (시작일 <= 오늘 <= 마감일)
        conditions.append({"reg_start_int": {"$lte": today_int}})
        conditions.append({"reg_end_int": {"$gte": today_int}})
    elif reg_status == "closing_soon":
        # 마감 7일 이내
        week_later = int((datetime.now() + timedelta(days=7)).strftime("%Y%m%d"))
        conditions.append({"reg_end_int": {"$gte": today_int}})
        conditions.append({"reg_end_int": {"$lte": week_later}})
    elif reg_status == "upcoming":
        # 등록 시작 전
        conditions.append({"reg_start_int": {"$gt": today_int}})
    elif reg_status == "exclude_closed":
        # 마감 안 된 것만
        conditions.append({"reg_end_int": {"$gte": today_int}})

    # 행사 기간 필터
    duration = parse_duration_filter(query)
    if duration == "multi_day":
        conditions.append({"duration_days": {"$gt": 1}})
    elif duration == "single_day":
        conditions.append({"duration_days": {"$eq": 1}})

    # 참고: 장소 필터는 ChromaDB에서 $contains 미지원으로 Python 후처리에서 수행

    if not conditions:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def filter_nodes_by_location(nodes: list, location: str) -> list:
    """Filter nodes by location (Python post-processing)."""
    if not location:
        return nodes

    filtered = []
    for node in nodes:
        metadata = node.metadata if hasattr(node, 'metadata') else {}
        node_location = metadata.get("location", "")
        if location in node_location:
            filtered.append(node)

    return filtered


def build_filter_description(query: str) -> str:
    """적용된 필터를 사람이 읽을 수 있는 형태로 설명."""
    descriptions = []

    year, month, month_range = parse_date_from_query(query)
    if year:
        descriptions.append(f"{year}년")
    if month:
        descriptions.append(f"{month}월")
    if month_range:
        descriptions.append(f"{min(month_range)}월~{max(month_range)}월")

    is_weekend = parse_weekend_filter(query)
    if is_weekend is True:
        descriptions.append("주말(토/일) 행사")
    elif is_weekend is False:
        descriptions.append("평일 행사")

    category = parse_category_from_query(query)
    if category:
        descriptions.append(f"카테고리: {category}")

    exclusion = parse_exclusion_filter(query)
    if exclusion:
        descriptions.append(f"{exclusion} 제외")

    reg_status = parse_registration_filter(query)
    if reg_status == "available":
        descriptions.append("현재 등록 가능")
    elif reg_status == "closing_soon":
        descriptions.append("등록 마감 임박")
    elif reg_status == "upcoming":
        descriptions.append("등록 시작 전")

    duration = parse_duration_filter(query)
    if duration == "multi_day":
        descriptions.append("며칠간 진행 행사")
    elif duration == "single_day":
        descriptions.append("당일 행사")

    location = parse_location_from_query(query)
    if location:
        descriptions.append(f"장소: {location}")

    if is_time_based_query(query):
        descriptions.append("오늘 이후 행사")

    if descriptions:
        return f"[적용된 필터: {', '.join(descriptions)}]"
    return ""


def sort_nodes_by_date(nodes: list, ascending: bool = True) -> list:
    """Sort nodes by start_date_int."""
    def get_date_int(node):
        metadata = node.metadata if hasattr(node, 'metadata') else {}
        return metadata.get("start_date_int", 99999999)  # 날짜 없으면 맨 뒤로

    return sorted(nodes, key=get_date_int, reverse=not ascending)


def calculate_registration_status(metadata: dict) -> str:
    """등록 상태를 실시간 계산."""
    today_int = int(datetime.now().strftime("%Y%m%d"))
    reg_start = metadata.get("reg_start_int")
    reg_end = metadata.get("reg_end_int")

    if not reg_start or not reg_end:
        return ""

    if today_int < reg_start:
        return "등록상태: 등록 시작 전"
    elif today_int <= reg_end:
        days_left = (reg_end % 100) - (today_int % 100)  # 간단한 일수 계산
        if reg_end // 100 != today_int // 100:  # 월이 다르면 대략 계산
            days_left = "며칠"
        if isinstance(days_left, int) and days_left <= 7:
            return f"등록상태: 등록 가능 (마감 {days_left}일 전)"
        return "등록상태: 등록 가능"
    else:
        return "등록상태: 등록 마감"


def format_nodes_as_context(nodes: list, max_nodes: int | None = None) -> str:
    """Format nodes as concise context string for LLM."""
    if max_nodes:
        nodes = nodes[:max_nodes]

    context_parts = []
    for i, node in enumerate(nodes, 1):
        metadata = node.metadata if hasattr(node, 'metadata') else {}

        # 간결한 포맷 사용
        answer = metadata.get("answer_template", "")
        url = metadata.get("url", "")

        # 등록 상태 실시간 계산
        reg_status = calculate_registration_status(metadata)

        if answer:
            entry = f"{i}. {answer}"
            if reg_status:
                entry += f"\n   {reg_status}"
            if url:
                entry += f"\n   URL: {url}"
            context_parts.append(entry)
        else:
            text = node.text if hasattr(node, 'text') else str(node)
            context_parts.append(f"{i}. {text[:300]}")

    return "\n\n".join(context_parts)


def chat(message: str) -> str:
    """
    Process a chat message and return the response.

    Args:
        message: User message

    Returns:
        AI response string
    """
    setup_settings()

    # 쿼리에서 필터 추출
    chroma_filters = build_chroma_filters(message)

    # 장소 필터 추출 (Python 후처리용)
    location = parse_location_from_query(message)

    if chroma_filters or location:
        if chroma_filters:
            # 필터가 있으면 ChromaDB에서 모든 매칭 문서 직접 조회
            nodes = get_all_by_filter(chroma_filters)
            print(f"[검색] 필터 적용: {chroma_filters}, 결과: {len(nodes)}개 (전체)")
        else:
            # 장소 필터만 있는 경우 전체 문서에서 필터링
            nodes = get_all_by_filter(None)
            print(f"[검색] 전체 문서 조회: {len(nodes)}개")

        # 장소 필터 (Python 후처리 - ChromaDB는 $contains 미지원)
        if location:
            nodes = filter_nodes_by_location(nodes, location)
            print(f"[필터] 장소 필터 적용: {location}, 결과: {len(nodes)}개")

        if not nodes:
            return "해당 조건에 맞는 문서를 찾을 수 없습니다."

        # 시간 기반 쿼리면 날짜순 정렬
        if is_time_based_query(message):
            nodes = sort_nodes_by_date(nodes, ascending=True)
            print(f"[정렬] 날짜순 정렬 완료")

        # 문서 수 제한 (LLM 속도 최적화)
        max_docs = config.RETRIEVAL_K  # 기본 20개
        display_count = min(max_docs, len(nodes))
        total_count = len(nodes)
        context = format_nodes_as_context(nodes, max_nodes=max_docs)
        print(f"[LLM] {display_count}개 문서 전달")

        # 적용된 필터 설명 생성
        filter_desc = build_filter_description(message)

        llm = get_llm()
        system_prompt = get_system_prompt()
        prompt = f"""{system_prompt}

{filter_desc}
다음은 질문 조건에 맞는 문서 {total_count}개 중 {display_count}개입니다:

{context}

위 문서들은 이미 질문 조건에 맞게 필터링된 결과입니다.
이 문서들을 바탕으로 답변하세요. 여러 개면 전부 나열하세요.
반드시 한국어로만 답변하세요.

질문: {message}

답변:"""

        response = llm.complete(prompt)
        return str(response)

    else:
        # 필터가 없으면 유사도 검색 사용
        index = get_index()
        query_engine = index.as_query_engine(
            similarity_top_k=config.RETRIEVAL_K,
            text_qa_template=get_qa_prompt(),
            system_prompt=get_system_prompt(),
        )

        response = query_engine.query(message)

        if hasattr(response, 'source_nodes'):
            print(f"[검색] 유사도 검색 결과: {len(response.source_nodes)}개")

        return str(response)


def reset_chat_engine() -> None:
    """Reset function (placeholder for compatibility)."""
    print("[초기화] ChatEngine 리셋 완료")
