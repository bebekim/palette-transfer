# ABOUTME: This file tests Reinhard color transfer algorithm for medical image standardization
# ABOUTME: Validates LAB color space transformations and statistical matching for hair clinic documentation

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'palette-transfer'))
from reinhard_transfer import ReinhardColorTransfer


class TestReinhardColorTransfer:
    """Test Reinhard color transfer for medical image standardization"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.source_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.target_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_initialization(self):
        """Test proper initialization"""
        reinhard = ReinhardColorTransfer(clip_values=True)
        
        assert reinhard.clip_values == True
        assert reinhard.source_mean is None
        assert reinhard.source_std is None
        
        reinhard_no_clip = ReinhardColorTransfer(clip_values=False)
        assert reinhard_no_clip.clip_values == False
    
    def test_preprocess_rgb_to_lab_conversion(self):
        """Test preprocessing converts RGB to LAB correctly"""
        reinhard = ReinhardColorTransfer()
        lab_image = reinhard._preprocess(self.source_image)
        
        assert lab_image.shape == self.source_image.shape
        assert lab_image.dtype == np.uint8
        
        # Verify it's actually converted to LAB (different from RGB)
        assert not np.array_equal(lab_image, self.source_image)
    
    def test_preprocess_invalid_channels(self):
        """Test rejection of images with incorrect channel count"""
        invalid_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        reinhard = ReinhardColorTransfer()
        
        with pytest.raises(AssertionError, match="image must have exactly 3 color channels"):
            reinhard._preprocess(invalid_image)
    
    def test_preprocess_invalid_dtype(self):
        """Test rejection of images with incorrect data type"""
        invalid_image = np.random.random((100, 100, 3)).astype(np.float32)
        reinhard = ReinhardColorTransfer()
        
        with pytest.raises(AssertionError, match="image must be in np.uint8 type"):
            reinhard._preprocess(invalid_image)
    
    def test_get_mean_std_calculation(self):
        """Test mean and standard deviation calculation for LAB channels"""
        reinhard = ReinhardColorTransfer()
        lab_image = reinhard._preprocess(self.source_image)
        
        mean, std = reinhard._get_mean_std(lab_image)
        
        assert len(mean) == 3  # L, a, b channels
        assert len(std) == 3
        assert all(s >= 0 for s in std)  # Standard deviation should be non-negative
    
    def test_fit_calculates_statistics(self):
        """Test fitting calculates proper LAB statistics"""
        reinhard = ReinhardColorTransfer()
        reinhard.fit(self.source_image)
        
        assert reinhard.source_mean is not None
        assert reinhard.source_std is not None
        assert len(reinhard.source_mean) == 3
        assert len(reinhard.source_std) == 3
        assert hasattr(reinhard, 'source_shape')
        assert reinhard.source_shape == self.source_image.shape
    
    def test_recolor_without_fit_raises_error(self):
        """Test recoloring without fitting raises appropriate error"""
        reinhard = ReinhardColorTransfer()
        
        with pytest.raises(ValueError, match="You must call fit\\(\\) before recolor\\(\\)"):
            reinhard.recolor(self.target_image)
    
    def test_recolor_output_validity(self):
        """Test recoloring produces valid medical image output"""
        reinhard = ReinhardColorTransfer()
        reinhard.fit(self.source_image)
        
        result = reinhard.recolor(self.target_image)
        
        assert result.shape == self.target_image.shape
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)
    
    def test_recolor_changes_image(self):
        """Test that recoloring actually changes the target image"""
        # Create images with different color characteristics
        source = np.full((50, 50, 3), [200, 100, 50], dtype=np.uint8)
        target = np.full((50, 50, 3), [50, 150, 200], dtype=np.uint8)
        
        reinhard = ReinhardColorTransfer()
        reinhard.fit(source)
        result = reinhard.recolor(target)
        
        # Result should be different from target
        assert not np.array_equal(result, target)
        assert result.shape == target.shape
    
    def test_zero_std_handling(self):
        """Test handling of zero standard deviation in medical images"""
        # Create image with constant color in one channel
        constant_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        constant_image[:, :, 0] = 100  # Constant L channel after conversion
        
        reinhard = ReinhardColorTransfer()
        reinhard.fit(self.source_image)
        
        # Should not raise error even with zero std
        result = reinhard.recolor(constant_image)
        assert result.shape == constant_image.shape
        assert result.dtype == np.uint8
    
    def test_clipping_behavior(self):
        """Test value clipping behavior for medical image safety"""
        reinhard_clip = ReinhardColorTransfer(clip_values=True)
        reinhard_no_clip = ReinhardColorTransfer(clip_values=False)
        
        reinhard_clip.fit(self.source_image)
        reinhard_no_clip.fit(self.source_image)
        
        result_clip = reinhard_clip.recolor(self.target_image)
        result_no_clip = reinhard_no_clip.recolor(self.target_image)
        
        # Clipped version should have values in valid range
        assert np.all(result_clip >= 0) and np.all(result_clip <= 255)
        
        # Both should have same shape and dtype
        assert result_clip.shape == result_no_clip.shape
        assert result_clip.dtype == result_no_clip.dtype == np.uint8
    
    def test_statistical_transfer_properties(self):
        """Test that color transfer actually transfers statistical properties"""
        # Create source with known statistical properties
        np.random.seed(42)
        source = np.random.normal(150, 30, (100, 100, 3)).astype(np.uint8)
        source = np.clip(source, 0, 255)
        
        target = np.random.normal(100, 50, (100, 100, 3)).astype(np.uint8)
        target = np.clip(target, 0, 255)
        
        reinhard = ReinhardColorTransfer()
        reinhard.fit(source)
        result = reinhard.recolor(target)
        
        # Result should be closer to source statistics than original target
        source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
        result_lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        
        source_mean = np.mean(source_lab.reshape(-1, 3), axis=0)
        result_mean = np.mean(result_lab.reshape(-1, 3), axis=0)
        
        # The means should be somewhat similar (allowing for conversion precision)
        for i in range(3):
            assert abs(source_mean[i] - result_mean[i]) < 50  # Reasonable tolerance


class TestMedicalComplianceFeatures:
    """Test medical compliance features for Reinhard color transfer"""
    
    def test_deterministic_results(self):
        """Test that results are deterministic for medical reproducibility"""
        source = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        # Create two identical instances
        reinhard1 = ReinhardColorTransfer()
        reinhard2 = ReinhardColorTransfer()
        
        # Fit both with same data
        reinhard1.fit(source)
        reinhard2.fit(source)
        
        # Results should be identical
        result1 = reinhard1.recolor(target)
        result2 = reinhard2.recolor(target)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_edge_case_grayscale_like_images(self):
        """Test handling of near-grayscale images (common in medical imaging)"""
        # Create near-grayscale image
        gray_values = np.random.randint(50, 200, (50, 50, 1), dtype=np.uint8)
        near_gray = np.repeat(gray_values, 3, axis=2)
        # Add small color variations
        near_gray[:, :, 1] += np.random.randint(-5, 6, (50, 50), dtype=np.int8)
        near_gray[:, :, 2] += np.random.randint(-5, 6, (50, 50), dtype=np.int8)
        near_gray = np.clip(near_gray, 0, 255).astype(np.uint8)
        
        reinhard = ReinhardColorTransfer()
        reinhard.fit(near_gray)
        
        result = reinhard.recolor(near_gray)
        
        assert result.shape == near_gray.shape
        assert result.dtype == np.uint8
    
    def test_large_image_processing(self):
        """Test processing of larger medical images"""
        large_source = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        large_target = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        reinhard = ReinhardColorTransfer()
        reinhard.fit(large_source)
        
        result = reinhard.recolor(large_target)
        
        assert result.shape == large_target.shape
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)
    
    def test_different_image_sizes(self):
        """Test that source and target images can have different sizes"""
        small_source = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        large_target = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        
        reinhard = ReinhardColorTransfer()
        reinhard.fit(small_source)
        
        result = reinhard.recolor(large_target)
        
        assert result.shape == large_target.shape
        assert result.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__])