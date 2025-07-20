# ABOUTME: This file tests targeted Reinhard color transfer for medical before/after images
# ABOUTME: Validates face detection, skin segmentation, and medical parameter validation for hair clinics

import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'palette-transfer'))
from targeted_transfer import TargetedReinhardTransfer, validate_blend_parameters


class TestParameterValidation:
    """Test medical parameter validation for targeted transfer"""
    
    def test_valid_blend_parameters(self):
        """Test validation of valid medical blend parameters"""
        # These should not raise any errors
        validate_blend_parameters(0.9, 0.5, 0.3)
        validate_blend_parameters(0.0, 0.0, 0.0)
        validate_blend_parameters(1.0, 1.0, 1.0)
        validate_blend_parameters(0.5, 0.5, 0.5)
    
    def test_invalid_skin_blend_parameter(self):
        """Test rejection of invalid skin blend parameters"""
        with pytest.raises(AssertionError, match="Skin blend must be between 0.0 and 1.0"):
            validate_blend_parameters(-0.1, 0.5, 0.3)
        
        with pytest.raises(AssertionError, match="Skin blend must be between 0.0 and 1.0"):
            validate_blend_parameters(1.1, 0.5, 0.3)
    
    def test_invalid_hair_blend_parameter(self):
        """Test rejection of invalid hair blend parameters"""
        with pytest.raises(AssertionError, match="Hair blend must be between 0.0 and 1.0"):
            validate_blend_parameters(0.9, -0.1, 0.3)
        
        with pytest.raises(AssertionError, match="Hair blend must be between 0.0 and 1.0"):
            validate_blend_parameters(0.9, 1.1, 0.3)
    
    def test_invalid_background_blend_parameter(self):
        """Test rejection of invalid background blend parameters"""
        with pytest.raises(AssertionError, match="Background blend must be between 0.0 and 1.0"):
            validate_blend_parameters(0.9, 0.5, -0.1)
        
        with pytest.raises(AssertionError, match="Background blend must be between 0.0 and 1.0"):
            validate_blend_parameters(0.9, 0.5, 1.1)


class TestTargetedReinhardTransfer:
    """Test targeted Reinhard transfer for medical imaging requirements"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.source_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        self.target_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    def test_initialization_with_medical_parameters(self):
        """Test initialization with medical-appropriate blend factors"""
        targeted = TargetedReinhardTransfer(
            skin_blend_factor=0.9,
            hair_region_blend_factor=0.5,
            background_blend_factor=0.3
        )
        
        assert targeted.skin_blend_factor == 0.9
        assert targeted.hair_region_blend_factor == 0.5
        assert targeted.background_blend_factor == 0.3
        assert targeted.face_cascade is not None
        assert targeted.reinhard is not None
    
    def test_initialization_with_defaults(self):
        """Test initialization with default medical parameters"""
        targeted = TargetedReinhardTransfer()
        
        assert targeted.skin_blend_factor == 0.9
        assert targeted.hair_region_blend_factor == 0.5
        assert targeted.background_blend_factor == 0.3
    
    def test_initialization_parameter_validation(self):
        """Test that initialization validates parameters"""
        with pytest.raises(AssertionError):
            TargetedReinhardTransfer(skin_blend_factor=-0.1)
        
        with pytest.raises(AssertionError):
            TargetedReinhardTransfer(hair_region_blend_factor=1.1)
        
        with pytest.raises(AssertionError):
            TargetedReinhardTransfer(background_blend_factor=-0.5)
    
    def test_skin_mask_creation_no_face(self):
        """Test skin mask creation when no face is detected"""
        targeted = TargetedReinhardTransfer()
        
        # Mock face detection to return no faces
        with patch.object(targeted.face_cascade, 'detectMultiScale') as mock_detect:
            mock_detect.return_value = np.array([])  # No faces detected
            
            mask = targeted._create_skin_mask(self.target_image)
            
            assert mask.shape == self.target_image.shape[:2]
            assert mask.dtype == np.float32
            assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
    
    def test_skin_mask_creation_with_face(self):
        """Test skin mask creation when face is detected"""
        targeted = TargetedReinhardTransfer()
        
        # Mock face detection to return a face
        with patch.object(targeted.face_cascade, 'detectMultiScale') as mock_detect:
            mock_detect.return_value = np.array([[50, 50, 100, 100]])  # Mock face detection
            
            mask = targeted._create_skin_mask(self.target_image)
            
            assert mask.shape == self.target_image.shape[:2]
            assert mask.dtype == np.float32
            assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
            
            # Hair region mask should also be created when face is detected
            assert targeted.hair_region_mask is not None
            assert targeted.hair_region_mask.shape == self.target_image.shape[:2]
    
    def test_fit_functionality(self):
        """Test fitting the underlying Reinhard transfer"""
        targeted = TargetedReinhardTransfer()
        targeted.fit(self.source_image)
        
        # Should delegate to the underlying Reinhard transfer
        assert targeted.reinhard.source_mean is not None
        assert targeted.reinhard.source_std is not None
    
    def test_recolor_with_medical_standards(self):
        """Test recoloring maintains medical image quality"""
        targeted = TargetedReinhardTransfer()
        targeted.fit(self.source_image)
        
        result = targeted.recolor(self.target_image)
        
        assert result.shape == self.target_image.shape
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)
    
    def test_recolor_creates_masks(self):
        """Test that recoloring creates and stores masks"""
        targeted = TargetedReinhardTransfer()
        targeted.fit(self.source_image)
        
        # Initially masks should be None
        assert targeted.skin_mask is None
        assert targeted.hair_region_mask is None
        
        result = targeted.recolor(self.target_image)
        
        # After recoloring, masks should be created
        assert targeted.skin_mask is not None
        assert targeted.hair_region_mask is not None
    
    def test_mask_storage_for_audit(self):
        """Test mask storage for medical audit requirements"""
        targeted = TargetedReinhardTransfer()
        targeted.fit(self.source_image)
        targeted.recolor(self.target_image)
        
        # Masks should be stored for medical audit trails
        assert targeted.skin_mask is not None
        assert targeted.hair_region_mask is not None
        assert targeted.skin_mask.shape == self.target_image.shape[:2]
        assert targeted.hair_region_mask.shape == self.target_image.shape[:2]
    
    def test_different_blend_factors_produce_different_results(self):
        """Test that different blend factors produce different medical outcomes"""
        source = np.full((100, 100, 3), [200, 100, 50], dtype=np.uint8)
        target = np.full((100, 100, 3), [50, 150, 200], dtype=np.uint8)
        
        # Test with high skin blend
        targeted_high = TargetedReinhardTransfer(skin_blend_factor=0.9)
        targeted_high.fit(source)
        result_high = targeted_high.recolor(target)
        
        # Test with low skin blend
        targeted_low = TargetedReinhardTransfer(skin_blend_factor=0.1)
        targeted_low.fit(source)
        result_low = targeted_low.recolor(target)
        
        # Results should be different
        assert not np.array_equal(result_high, result_low)
        assert result_high.shape == result_low.shape
    
    def test_hair_region_mask_fallback(self):
        """Test hair region mask creation when no face is detected"""
        targeted = TargetedReinhardTransfer()
        
        # Mock face detection to return no faces
        with patch.object(targeted.face_cascade, 'detectMultiScale') as mock_detect:
            mock_detect.return_value = np.array([])  # No faces detected
            
            targeted.fit(self.source_image)
            result = targeted.recolor(self.target_image)
            
            # Hair region mask should still be created as fallback
            assert targeted.hair_region_mask is not None
            assert targeted.hair_region_mask.shape == self.target_image.shape[:2]
            
            # Should use top 25% of image as approximation
            hair_height = self.target_image.shape[0] // 4
            top_region = targeted.hair_region_mask[:hair_height, :]
            bottom_region = targeted.hair_region_mask[hair_height:, :]
            
            # Top region should have higher values than bottom
            assert np.mean(top_region) > np.mean(bottom_region)


class TestMedicalComplianceFeatures:
    """Test medical compliance and safety features for targeted transfer"""
    
    def test_parameter_bounds_enforcement(self):
        """Test that parameters are strictly enforced for medical safety"""
        # Test edge cases at boundaries
        targeted_min = TargetedReinhardTransfer(0.0, 0.0, 0.0)
        assert targeted_min.skin_blend_factor == 0.0
        
        targeted_max = TargetedReinhardTransfer(1.0, 1.0, 1.0)
        assert targeted_max.skin_blend_factor == 1.0
    
    def test_deterministic_mask_creation(self):
        """Test that mask creation is deterministic for medical reproducibility"""
        targeted1 = TargetedReinhardTransfer()
        targeted2 = TargetedReinhardTransfer()
        
        # Mock face detection to be consistent
        with patch.object(targeted1.face_cascade, 'detectMultiScale') as mock1, \
             patch.object(targeted2.face_cascade, 'detectMultiScale') as mock2:
            
            mock1.return_value = np.array([[50, 50, 100, 100]])
            mock2.return_value = np.array([[50, 50, 100, 100]])
            
            mask1 = targeted1._create_skin_mask(self.target_image)
            mask2 = targeted2._create_skin_mask(self.target_image)
            
            # Masks should be identical
            np.testing.assert_array_almost_equal(mask1, mask2)
    
    def test_processing_safety_with_extreme_images(self):
        """Test safe processing of extreme medical image conditions"""
        # Test with very bright image
        bright_image = np.full((100, 100, 3), 250, dtype=np.uint8)
        
        # Test with very dark image
        dark_image = np.full((100, 100, 3), 5, dtype=np.uint8)
        
        targeted = TargetedReinhardTransfer()
        targeted.fit(bright_image)
        
        result_dark = targeted.recolor(dark_image)
        result_bright = targeted.recolor(bright_image)
        
        # Both should produce valid results
        assert result_dark.shape == dark_image.shape
        assert result_bright.shape == bright_image.shape
        assert np.all(result_dark >= 0) and np.all(result_dark <= 255)
        assert np.all(result_bright >= 0) and np.all(result_bright <= 255)
    
    def test_audit_trail_preservation(self):
        """Test that all processing steps preserve audit information"""
        targeted = TargetedReinhardTransfer()
        targeted.fit(self.source_image)
        
        # Process multiple images to ensure audit trail is maintained
        results = []
        for i in range(3):
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = targeted.recolor(test_image)
            results.append(result)
            
            # Audit information should be preserved after each processing
            assert targeted.skin_mask is not None
            assert targeted.hair_region_mask is not None
        
        # All results should be valid
        for result in results:
            assert result.dtype == np.uint8
            assert np.all(result >= 0) and np.all(result <= 255)


class TestIntegrationWithReinhardTransfer:
    """Test integration between targeted transfer and underlying Reinhard algorithm"""
    
    def test_reinhard_integration(self):
        """Test that targeted transfer properly uses Reinhard algorithm"""
        targeted = TargetedReinhardTransfer()
        
        # Verify that underlying Reinhard is properly initialized
        assert hasattr(targeted, 'reinhard')
        assert targeted.reinhard is not None
        
        # Test that fit delegates to Reinhard
        targeted.fit(self.source_image)
        assert targeted.reinhard.source_mean is not None
        assert targeted.reinhard.source_std is not None
    
    def test_full_pipeline_integration(self):
        """Test complete pipeline from fit to recolor"""
        source = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        
        targeted = TargetedReinhardTransfer(
            skin_blend_factor=0.8,
            hair_region_blend_factor=0.6,
            background_blend_factor=0.4
        )
        
        # Complete pipeline
        targeted.fit(source)
        result = targeted.recolor(target)
        
        # Verify complete processing
        assert result.shape == target.shape
        assert result.dtype == np.uint8
        assert not np.array_equal(result, target)  # Should actually process
        assert targeted.skin_mask is not None
        assert targeted.hair_region_mask is not None


if __name__ == "__main__":
    pytest.main([__file__])