# ABOUTME: Transfer service - orchestrates palette transfer workflows
# ABOUTME: Coordinates between algorithms, storage, and job persistence

import time
from uuid import UUID

import numpy as np

from app.domain.transfer import (
    TransferMethod,
    TransferParams,
    TransferJob,
    TransferResult,
    TransferStatus,
    SkinDetectionParams,
    SkinDetectionResult,
)
from app.domain.exceptions import ImageProcessingError
from app.services.ports import TransferJobRepository, ImageStorage
from app.services.algorithms import (
    SkinToneTransfer,
    transfer_reinhard,
    transfer_hybrid,
    transfer_optimized,
)


class TransferService:
    """Orchestrates palette transfer operations."""

    def __init__(
        self,
        job_repository: TransferJobRepository | None = None,
        image_storage: ImageStorage | None = None,
    ):
        self.job_repository = job_repository
        self.image_storage = image_storage

    def detect_skin(
        self,
        image: np.ndarray,
        params: SkinDetectionParams | None = None,
    ) -> SkinDetectionResult:
        """Detect skin regions in an image.

        Args:
            image: RGB image as numpy array
            params: Skin detection parameters

        Returns:
            SkinDetectionResult with mask and statistics
        """
        if params is None:
            params = SkinDetectionParams()

        transfer = SkinToneTransfer(
            skin_ycrcb_lower=(0, params.cr_low, params.cb_low),
            skin_ycrcb_upper=(255, params.cr_high, params.cb_high),
        )

        skin_mask, _ = transfer._create_skin_mask(image)

        skin_pixels = int(np.sum(skin_mask > 0.5))
        total_pixels = image.shape[0] * image.shape[1]

        # Create visualization overlay
        viz = image.copy().astype(np.float32)
        viz[skin_mask > 0.5] = viz[skin_mask > 0.5] * 0.5 + np.array([255, 100, 100]) * 0.5
        viz = np.clip(viz, 0, 255).astype(np.uint8)

        return SkinDetectionResult(
            skin_pixels=skin_pixels,
            total_pixels=total_pixels,
            skin_percentage=round(skin_pixels / total_pixels * 100, 2),
            visualization_base64=self._image_to_base64(viz),
            skin_mask_base64=self._image_to_base64((skin_mask * 255).astype(np.uint8)),
            detection_params=params,
        )

    def execute_transfer(
        self,
        source: np.ndarray,
        target: np.ndarray,
        params: TransferParams,
        user_id: UUID | None = None,
    ) -> TransferResult:
        """Execute palette transfer from source to target.

        Args:
            source: Source RGB image (palette to extract)
            target: Target RGB image (to recolor)
            params: Transfer parameters
            user_id: Optional user ID for job tracking

        Returns:
            TransferResult with processed image
        """
        start_time = time.time()

        # Create job record if we have a repository and user
        job = None
        if self.job_repository and user_id:
            job = TransferJob(user_id=user_id, params=params, status=TransferStatus.PROCESSING)
            job = self.job_repository.create(job)

        try:
            result_image = self._apply_transfer(source, target, params)
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Get skin pixel counts for skintone method
            source_skin_pixels = None
            target_skin_pixels = None

            if params.method == TransferMethod.SKINTONE:
                transfer = SkinToneTransfer(
                    skin_ycrcb_lower=(0, params.skin_detection.cr_low, params.skin_detection.cb_low),
                    skin_ycrcb_upper=(255, params.skin_detection.cr_high, params.skin_detection.cb_high),
                )
                transfer.fit(source)
                transfer.recolor(target)
                source_skin_pixels = int(np.sum(transfer.source_skin_mask > 0.5))
                target_skin_pixels = int(np.sum(transfer.target_skin_mask > 0.5))

            # Update job if tracking
            if job and self.job_repository:
                job.status = TransferStatus.COMPLETED
                job.processing_time_ms = processing_time_ms
                job = self.job_repository.update(job)

            return TransferResult(
                job=job or TransferJob(user_id=user_id or UUID(int=0), params=params, status=TransferStatus.COMPLETED, processing_time_ms=processing_time_ms),
                result_base64=self._image_to_base64(result_image),
                source_skin_pixels=source_skin_pixels,
                target_skin_pixels=target_skin_pixels,
            )

        except Exception as e:
            if job and self.job_repository:
                job.status = TransferStatus.FAILED
                job.error_message = str(e)
                self.job_repository.update(job)
            raise ImageProcessingError(f"Transfer failed: {e}") from e

    def compare_methods(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> dict[str, TransferResult | str]:
        """Compare all transfer methods side by side.

        Args:
            source: Source RGB image
            target: Target RGB image

        Returns:
            Dictionary mapping method name to result or error message
        """
        results = {}

        for method in TransferMethod:
            try:
                params = TransferParams(method=method)
                result = self.execute_transfer(source, target, params)
                results[method.value] = result
            except Exception as e:
                results[method.value] = f"Error: {e}"

        return results

    def _apply_transfer(
        self,
        source: np.ndarray,
        target: np.ndarray,
        params: TransferParams,
    ) -> np.ndarray:
        """Apply the appropriate transfer algorithm.

        Args:
            source: Source RGB image
            target: Target RGB image
            params: Transfer parameters

        Returns:
            Processed RGB image
        """
        if params.method == TransferMethod.SKINTONE:
            transfer = SkinToneTransfer(
                skin_blend_factor=params.skin_blend,
                hair_region_blend_factor=params.hair_blend,
                background_blend_factor=params.bg_blend,
                skin_ycrcb_lower=(0, params.skin_detection.cr_low, params.skin_detection.cb_low),
                skin_ycrcb_upper=(255, params.skin_detection.cr_high, params.skin_detection.cb_high),
            )
            transfer.fit(source)
            return transfer.recolor(target)

        elif params.method == TransferMethod.REINHARD:
            return transfer_reinhard(target, source, preserve_luminance=params.preserve_luminance)

        elif params.method == TransferMethod.HYBRID:
            return transfer_hybrid(target, source, skin_weight=params.skin_weight)

        elif params.method == TransferMethod.OPTIMIZED:
            return transfer_optimized(target, source, preserve_luminance=params.preserve_luminance)

        else:
            raise ValueError(f"Unknown method: {params.method}")

    def _image_to_base64(self, image: np.ndarray, format: str = "PNG") -> str:
        """Convert numpy array to base64 data URL."""
        import base64
        from io import BytesIO
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image.astype(np.uint8))
        buffer = BytesIO()
        pil_image.save(buffer, format=format, quality=90)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{b64}"
