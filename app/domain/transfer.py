# ABOUTME: Transfer domain entities - business rules for palette transfer operations
# ABOUTME: Includes validation rules for algorithm parameters

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TransferMethod(str, Enum):
    """Available transfer algorithms."""

    SKINTONE = "skintone"
    REINHARD = "reinhard"
    HYBRID = "hybrid"
    OPTIMIZED = "optimized"


class TransferStatus(str, Enum):
    """Transfer job status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SkinDetectionParams(BaseModel):
    """Parameters for skin detection in YCrCb color space."""

    cr_low: int = Field(default=133, ge=0, le=255)
    cr_high: int = Field(default=173, ge=0, le=255)
    cb_low: int = Field(default=77, ge=0, le=255)
    cb_high: int = Field(default=127, ge=0, le=255)


class TransferParams(BaseModel):
    """Parameters for palette transfer operation."""

    method: TransferMethod = TransferMethod.SKINTONE

    # Blend factors (0.0 = keep original, 1.0 = full transfer)
    skin_blend: float = Field(default=0.9, ge=0.0, le=1.0)
    hair_blend: float = Field(default=0.5, ge=0.0, le=1.0)
    bg_blend: float = Field(default=0.3, ge=0.0, le=1.0)

    # Skin detection bounds
    skin_detection: SkinDetectionParams = Field(default_factory=SkinDetectionParams)

    # Method-specific options
    preserve_luminance: bool = False
    skin_weight: float = Field(default=0.7, ge=0.0, le=1.0)  # For hybrid method


class TransferJob(BaseModel):
    """Record of a transfer operation."""

    id_: UUID = Field(default_factory=uuid4)
    user_id: UUID

    params: TransferParams
    status: TransferStatus = TransferStatus.PENDING
    error_message: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    processing_time_ms: int | None = None

    class Config:
        from_attributes = True


class SkinDetectionResult(BaseModel):
    """Result of skin detection operation."""

    skin_pixels: int
    total_pixels: int
    skin_percentage: float
    visualization_base64: str
    skin_mask_base64: str
    detection_params: SkinDetectionParams


class TransferResult(BaseModel):
    """Result of transfer operation."""

    job: TransferJob
    result_base64: str
    source_skin_pixels: int | None = None
    target_skin_pixels: int | None = None
