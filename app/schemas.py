"""Pydantic models: LLM structured-output schema and API request/response contracts."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# LLM structured-output schema
# The is_applicable flag distinguishes a genuine zero value from an
# explicitly non-applicable (N/A) line item.
# ---------------------------------------------------------------------------
class ExtractedItem(BaseModel):
    item_number: str = Field(description="Line item number")
    item: str = Field(description="Line item description")
    is_applicable: bool = Field(
        description=(
            "True if the item has a numerical value (including zero). "
            "False if the item is marked as 'N/A', 'Not applicable', "
            "'—', '-', or is explicitly stated as not applicable."
        )
    )
    value_1: float = Field(description="Primary value extracted. Use 0.0 if null or not applicable.")
    value_2: float = Field(description="Secondary value extracted. Use 0.0 if null or not applicable.")
    value_3: float = Field(description="Tertiary value extracted. Use 0.0 if null or not applicable.")


class FinancialExtraction(BaseModel):
    bank_name: str = Field(description="The name of the bank")
    template: str = Field(description="The template name (e.g., OV1, LR2)")
    year: str = Field(description="The financial year")
    unit: str = Field(description="The exact unit of measurement as found in the source text. Do not guess.")
    data: list[ExtractedItem] = Field(description="List of extracted line items")


# ---------------------------------------------------------------------------
# API contracts
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    bank_name: str = Field(examples=["EXAMPLE BANK (HONG KONG) LIMITED"])
    item: str = Field(examples=["Credit risk (excluding counterparty credit risk)"])
    template: str = Field(default="OV1", examples=["OV1"])
    year: str = Field(default="2024")
    strategy: Literal["sparse", "dense", "hybrid_rrf"] = "hybrid_rrf"
    top_k: int = Field(default=3, ge=1, le=10)


class SourceChunk(BaseModel):
    source: str
    preview: str


class QueryResponse(BaseModel):
    bank_name: str
    strategy: str
    query: str
    extraction: Optional[FinancialExtraction]
    sources: list[SourceChunk]