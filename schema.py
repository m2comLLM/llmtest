"""RAG document schema definitions using Pydantic."""

from typing import Optional
from pydantic import BaseModel, Field


class ContentSchema(BaseModel):
    """Content fields for RAG document."""

    question: str = Field(description="Expected question for search optimization")
    answer: str = Field(description="Answer template or response")
    explanation: Optional[str] = Field(default=None, description="Additional explanation")


class MetadataSchema(BaseModel):
    """Metadata fields for RAG document."""

    event_name: Optional[str] = Field(default=None, description="Event name")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    registration_start: Optional[str] = Field(default=None, description="Registration start")
    registration_end: Optional[str] = Field(default=None, description="Registration deadline")
    location: Optional[str] = Field(default=None, description="Event location")
    credits: Optional[str] = Field(default=None, description="Credits/points")
    url: Optional[str] = Field(default=None, description="Event URL")
    category: Optional[str] = Field(default=None, description="Event category")
    source_file: Optional[str] = Field(default=None, description="Source file path")
    row: Optional[int] = Field(default=None, description="Row number in source file")


class SearchBoostSchema(BaseModel):
    """Search boost fields for filtering."""

    year: Optional[int] = Field(default=None, description="Year for filtering")
    month: Optional[int] = Field(default=None, description="Month for filtering")
    day: Optional[int] = Field(default=None, description="Day for filtering")
    location_normalized: Optional[str] = Field(default=None, description="Normalized location")


class RAGDocument(BaseModel):
    """Complete RAG document schema."""

    id: str = Field(description="Unique document identifier")
    type: str = Field(default="event", description="Document type")
    content: ContentSchema = Field(description="Content fields")
    keywords: list[str] = Field(default_factory=list, description="Keywords for search")
    metadata: MetadataSchema = Field(default_factory=MetadataSchema, description="Metadata")
    search_boost: SearchBoostSchema = Field(
        default_factory=SearchBoostSchema, description="Search boost fields"
    )


# Keyword synonyms mapping for common medical/academic terms
KEYWORD_SYNONYMS: dict[str, list[str]] = {
    "COPD": ["만성폐쇄성폐질환", "만성 폐쇄성 폐질환", "chronic obstructive pulmonary disease"],
    "천식": ["asthma", "기관지천식"],
    "ILD": ["간질성폐질환", "interstitial lung disease"],
    "NTM": ["비결핵항산균", "nontuberculous mycobacteria"],
    "폐암": ["lung cancer", "폐암"],
    "결핵": ["TB", "tuberculosis"],
    "수면무호흡": ["sleep apnea", "수면호흡장애"],
    "폐기능": ["pulmonary function", "PFT"],
}


def get_synonyms(keyword: str) -> list[str]:
    """Get synonyms for a keyword."""
    synonyms = [keyword]

    # Check if keyword is in synonyms map
    keyword_lower = keyword.lower()
    for key, values in KEYWORD_SYNONYMS.items():
        if keyword_lower == key.lower() or keyword_lower in [v.lower() for v in values]:
            synonyms.extend([key] + values)
            break

    return list(set(synonyms))


def expand_keywords(keywords: list[str]) -> list[str]:
    """Expand keywords with synonyms."""
    expanded = []
    for keyword in keywords:
        expanded.extend(get_synonyms(keyword))
    return list(set(expanded))
