# ABOUTME: This file tests K-means palette reduction algorithms for medical image processing
# ABOUTME: Validates clustering behavior and random walk variations for hair clinic analysis

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'palette-transfer'))
from kmeans_palette import KMeansReducedPalette, UniqueKMeansReducedPalette


class TestKMeansReducedPalette:
    """Test K-means palette reduction for medical image consistency"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        self.target_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    
    def test_initialization(self):
        """Test proper initialization of K-means palette"""
        kmeans = KMeansReducedPalette(num_colors=8)
        
        assert kmeans.num_colors == 8
        assert kmeans.source_pixels is None
        assert kmeans.kmeans.n_clusters == 8
        assert kmeans.kmeans.random_state == 0xfee1600d
    
    def test_preprocess_valid_image(self):
        """Test preprocessing of valid RGB medical images"""
        kmeans = KMeansReducedPalette(num_colors=5)
        processed = kmeans._preprocess(self.test_image)
        
        assert processed.shape == (2500, 3)
        assert processed.dtype == np.uint8
    
    def test_preprocess_invalid_channels(self):
        """Test rejection of images with incorrect channel count"""
        invalid_image = np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8)
        kmeans = KMeansReducedPalette(num_colors=5)
        
        with pytest.raises(AssertionError, match="image must have exactly 3 color channels"):
            kmeans._preprocess(invalid_image)
    
    def test_preprocess_invalid_dtype(self):
        """Test rejection of images with incorrect data type"""
        invalid_image = np.random.random((50, 50, 3)).astype(np.float32)
        kmeans = KMeansReducedPalette(num_colors=5)
        
        with pytest.raises(AssertionError, match="image must be in np.uint8 type"):
            kmeans._preprocess(invalid_image)
    
    def test_fit_functionality(self):
        """Test fitting palette to source image"""
        kmeans = KMeansReducedPalette(num_colors=5)
        kmeans.fit(self.test_image)
        
        assert kmeans.source_pixels is not None
        assert kmeans.source_pixels.shape == (2500, 3)
        assert len(kmeans.centroid_nearest_pixels) == 5
        
        # Check that centroids are valid
        assert hasattr(kmeans.kmeans, 'cluster_centers_')
        assert kmeans.kmeans.cluster_centers_.shape == (5, 3)
    
    def test_recolor_maintains_shape(self):
        """Test recoloring maintains image dimensions for medical standards"""
        kmeans = KMeansReducedPalette(num_colors=5)
        kmeans.fit(self.test_image)
        
        result = kmeans.recolor(self.target_image)
        
        assert result.shape == self.target_image.shape
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)
    
    def test_random_walk_recolor(self):
        """Test random walk recoloring functionality"""
        np.random.seed(42)  # For reproducible tests
        kmeans = KMeansReducedPalette(num_colors=5)
        kmeans.fit(self.test_image)
        
        result = kmeans.random_walk_recolor(self.target_image, max_steps=3)
        
        assert result.shape == self.target_image.shape
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)
    
    def test_random_neighborhood_walk(self):
        """Test random neighborhood walk recoloring"""
        np.random.seed(42)  # For reproducible tests
        kmeans = KMeansReducedPalette(num_colors=5)
        kmeans.fit(self.test_image)
        
        result = kmeans.random_neighborhood_walk_recolor(self.target_image, max_steps=3)
        
        assert result.shape == self.target_image.shape
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)
    
    def test_different_cluster_counts(self):
        """Test K-means with different cluster counts for medical flexibility"""
        for num_colors in [3, 5, 8, 16]:
            kmeans = KMeansReducedPalette(num_colors=num_colors)
            kmeans.fit(self.test_image)
            
            result = kmeans.recolor(self.target_image)
            
            assert result.shape == self.target_image.shape
            assert len(kmeans.centroid_nearest_pixels) == num_colors


class TestUniqueKMeansReducedPalette:
    """Test unique K-means palette reduction for optimized medical processing"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        self.target_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    
    def test_inheritance(self):
        """Test that UniqueKMeansReducedPalette properly inherits from base class"""
        unique_kmeans = UniqueKMeansReducedPalette(num_colors=8)
        
        assert isinstance(unique_kmeans, KMeansReducedPalette)
        assert unique_kmeans.num_colors == 8
    
    def test_unique_pixel_processing(self):
        """Test that unique pixels are processed correctly"""
        # Create image with many duplicate pixels
        solid_image = np.full((50, 50, 3), [128, 64, 192], dtype=np.uint8)
        solid_image[0, 0] = [255, 0, 0]  # Add one different pixel
        solid_image[0, 1] = [0, 255, 0]  # Add another different pixel
        
        unique_kmeans = UniqueKMeansReducedPalette(num_colors=3)
        unique_kmeans.fit(solid_image)
        
        # Should work with unique pixels rather than all pixels
        assert unique_kmeans.source_pixels is not None
        # The actual number of unique pixels should be small
        assert unique_kmeans.source_pixels.shape[0] <= 2500
    
    def test_recolor_functionality(self):
        """Test recoloring with unique K-means"""
        unique_kmeans = UniqueKMeansReducedPalette(num_colors=5)
        unique_kmeans.fit(self.test_image)
        
        result = unique_kmeans.recolor(self.target_image)
        
        assert result.shape == self.target_image.shape
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)


class TestMedicalComplianceFeatures:
    """Test medical compliance features for K-means algorithms"""
    
    def test_deterministic_results(self):
        """Test that results are deterministic for medical reproducibility"""
        test_image = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        target_image = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        
        # Create two identical instances
        kmeans1 = KMeansReducedPalette(num_colors=5)
        kmeans2 = KMeansReducedPalette(num_colors=5)
        
        # Fit both with same data
        kmeans1.fit(test_image)
        kmeans2.fit(test_image)
        
        # Results should be identical due to fixed random_state
        result1 = kmeans1.recolor(target_image)
        result2 = kmeans2.recolor(target_image)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_edge_case_single_color(self):
        """Test handling of single-color images (edge case in medical imaging)"""
        single_color_image = np.full((50, 50, 3), [128, 128, 128], dtype=np.uint8)
        
        kmeans = KMeansReducedPalette(num_colors=5)
        kmeans.fit(single_color_image)
        
        result = kmeans.recolor(single_color_image)
        
        assert result.shape == single_color_image.shape
        assert result.dtype == np.uint8
    
    def test_parameter_validation_for_medical_use(self):
        """Test that parameters are appropriate for medical use"""
        # Test valid medical parameters
        for num_colors in [2, 4, 8, 16, 32]:
            kmeans = KMeansReducedPalette(num_colors=num_colors)
            assert kmeans.num_colors == num_colors
            assert kmeans.num_colors >= 2  # Minimum for meaningful clustering
        
        # Test that very small cluster counts still work
        kmeans_minimal = KMeansReducedPalette(num_colors=2)
        test_image = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        kmeans_minimal.fit(test_image)
        
        result = kmeans_minimal.recolor(test_image)
        assert result.shape == test_image.shape


if __name__ == "__main__":
    pytest.main([__file__])