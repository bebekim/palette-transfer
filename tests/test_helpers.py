# ABOUTME: This file tests helper functions for medical image processing and argument parsing
# ABOUTME: Validates image loading, color analysis, and CLI argument handling for hair clinic workflows

import pytest
import numpy as np
import tempfile
import os
import shutil
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'palette-transfer'))
from helpers import (
    get_image, get_unique_colors, get_image_stats, 
    visualize_palette, build_argument_parser,
    create_folder, copy_files_to_temp_folder, delete_folder_and_contents,
    get_relevant_filepaths, closest_rect, invert_image, split_image
)


class TestImageProcessingHelpers:
    """Test image processing helper functions for medical standards"""
    
    def setup_method(self):
        """Setup test fixtures with temporary image files"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_image.jpg")
        
        # Create and save test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(self.test_image_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
    
    def teardown_method(self):
        """Cleanup temporary files"""
        shutil.rmtree(self.temp_dir)
    
    def test_get_image_rgb_conversion(self):
        """Test image loading with RGB color space for medical processing"""
        image = get_image(self.test_image_path, color_space="RGB")
        
        assert image.shape[2] == 3
        assert image.dtype == np.uint8
        assert np.all(image >= 0) and np.all(image <= 255)
    
    def test_get_image_bgr_default(self):
        """Test image loading with default BGR color space"""
        image = get_image(self.test_image_path, color_space="BGR")
        
        assert image.shape[2] == 3
        assert image.dtype == np.uint8
        assert np.all(image >= 0) and np.all(image <= 255)
    
    def test_get_image_lab_conversion(self):
        """Test image loading with LAB color space for medical analysis"""
        image = get_image(self.test_image_path, color_space="LAB")
        
        assert image.shape[2] == 3
        assert image.dtype == np.uint8
    
    def test_get_image_hsv_conversion(self):
        """Test image loading with HSV color space"""
        image = get_image(self.test_image_path, color_space="HSV")
        
        assert image.shape[2] == 3
        assert image.dtype == np.uint8
    
    def test_get_image_invalid_colorspace(self):
        """Test error handling for invalid color space"""
        with pytest.raises(ValueError, match="Invalid color space provided"):
            get_image(self.test_image_path, color_space="INVALID")
    
    def test_get_unique_colors_analysis(self):
        """Test unique color extraction for medical image analysis"""
        colors = get_unique_colors(self.test_image_path, color_space="RGB")
        
        assert isinstance(colors, dict)
        assert len(colors) > 0
        
        # Check that all colors are valid RGB tuples
        for color, count in colors.items():
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
            assert count > 0
            assert isinstance(count, (int, np.integer))
    
    def test_get_image_stats_medical_metrics(self):
        """Test image statistics calculation for medical documentation"""
        stats = get_image_stats(self.test_image_path, color_space="RGB")
        
        height, width, num_pixels, num_unique_colors, ch1_mean, ch1_std, ch2_mean, ch2_std, ch3_mean, ch3_std = stats
        
        assert height > 0 and width > 0
        assert num_pixels == height * width
        assert num_unique_colors > 0
        assert 0 <= ch1_mean <= 255 and ch1_std >= 0
        assert 0 <= ch2_mean <= 255 and ch2_std >= 0
        assert 0 <= ch3_mean <= 255 and ch3_std >= 0
    
    def test_visualize_palette_functionality(self):
        """Test palette visualization for medical documentation"""
        # Create test palette
        test_palette = np.random.randint(0, 255, (20, 3), dtype=np.uint8)
        
        result = visualize_palette(test_palette, scale=0)
        
        # Should return PIL Image
        assert hasattr(result, 'size')
        assert hasattr(result, 'mode')
        
        # Test with scaling
        scaled_result = visualize_palette(test_palette, scale=2)
        assert scaled_result.size[0] >= result.size[0]
        assert scaled_result.size[1] >= result.size[1]


class TestFileManagementHelpers:
    """Test file management helper functions for medical image workflows"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_folder = os.path.join(self.temp_dir, "test_folder")
    
    def teardown_method(self):
        """Cleanup temporary files"""
        shutil.rmtree(self.temp_dir)
    
    def test_create_folder_new(self):
        """Test creating new folder for medical image organization"""
        create_folder(self.test_folder)
        
        assert os.path.exists(self.test_folder)
        assert os.path.isdir(self.test_folder)
    
    def test_create_folder_existing(self):
        """Test recreating existing folder (should replace)"""
        # Create folder first
        os.makedirs(self.test_folder)
        test_file = os.path.join(self.test_folder, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        # Recreate folder
        create_folder(self.test_folder)
        
        assert os.path.exists(self.test_folder)
        assert not os.path.exists(test_file)  # Should be removed
    
    def test_copy_files_to_temp_folder(self):
        """Test copying medical image files to temporary processing folder"""
        # Create test files
        file1 = os.path.join(self.temp_dir, "source.jpg")
        file2 = os.path.join(self.temp_dir, "target.jpg")
        
        test_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        cv2.imwrite(file1, test_img)
        cv2.imwrite(file2, test_img)
        
        folder_path, folder_name = copy_files_to_temp_folder(file1, file2)
        
        assert os.path.exists(folder_path)
        assert len(folder_name) == 10  # Random folder name length
        
        # Check that files were copied with correct naming
        src_file = os.path.join(folder_path, f"src_{os.path.basename(file1)}")
        tgt_file = os.path.join(folder_path, f"tgt_{os.path.basename(file2)}")
        
        assert os.path.exists(src_file)
        assert os.path.exists(tgt_file)
        
        # Cleanup
        delete_folder_and_contents(folder_path)
    
    def test_delete_folder_and_contents(self):
        """Test safe deletion of temporary processing folders"""
        test_folder = os.path.join(self.temp_dir, "to_delete")
        os.makedirs(test_folder)
        
        test_file = os.path.join(test_folder, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        delete_folder_and_contents(test_folder)
        
        assert not os.path.exists(test_folder)
    
    def test_get_relevant_filepaths(self):
        """Test getting relevant medical image file paths"""
        # Create test directory with various files
        test_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(test_dir)
        
        # Create test files
        jpg_file = os.path.join(test_dir, "image1.jpg")
        png_file = os.path.join(test_dir, "image2.png")
        txt_file = os.path.join(test_dir, "not_image.txt")
        
        for file_path in [jpg_file, png_file, txt_file]:
            with open(file_path, 'w') as f:
                f.write("test")
        
        # Test with image formats
        acceptable_formats = ['.jpg', '.png']
        relevant_files = get_relevant_filepaths(test_dir, acceptable_formats)
        
        assert len(relevant_files) == 2
        assert jpg_file in relevant_files
        assert png_file in relevant_files
        assert txt_file not in relevant_files


class TestImageUtilities:
    """Test image utility functions for medical processing"""
    
    def test_closest_rect_calculation(self):
        """Test calculation of closest rectangle dimensions"""
        # Test various pixel counts
        test_cases = [
            (100, (8, 16)),  # Should find closest 2:1 rectangle
            (50, (6, 12)),
            (200, (10, 20)),
        ]
        
        for n_pixels, expected in test_cases:
            k, w = closest_rect(n_pixels)
            assert w == 2 * k  # Should maintain 2:1 ratio
            assert 2 * k ** 2 >= n_pixels  # Should accommodate all pixels
    
    def test_invert_image_functionality(self):
        """Test image inversion for medical analysis"""
        test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        # Test horizontal flip (axis=1)
        inverted_h = invert_image(test_image, axis=1)
        assert inverted_h.shape == test_image.shape
        assert not np.array_equal(inverted_h, test_image)
        
        # Test vertical flip (axis=0)
        inverted_v = invert_image(test_image, axis=0)
        assert inverted_v.shape == test_image.shape
        assert not np.array_equal(inverted_v, test_image)
        
        # Double flip should return to original
        double_flip = invert_image(inverted_h, axis=1)
        np.testing.assert_array_equal(double_flip, test_image)
    
    def test_split_image_functionality(self):
        """Test image splitting for medical region analysis"""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test splitting into 2x2 tiles
            split_image(test_image, "test_image", (2, 2), temp_dir)
            
            # Should create 4 tiles
            expected_files = [
                "test_image_0_0.jpg",
                "test_image_0_1.jpg", 
                "test_image_1_0.jpg",
                "test_image_1_1.jpg"
            ]
            
            for filename in expected_files:
                filepath = os.path.join(temp_dir, filename)
                assert os.path.exists(filepath)
                
                # Verify tile is valid image
                tile = cv2.imread(filepath)
                assert tile is not None
                assert tile.shape[0] == 50  # Height should be half
                assert tile.shape[1] == 50  # Width should be half
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_split_image_return_specific_tile(self):
        """Test returning specific tile from image splitting"""
        test_image = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Request specific tile (1, 1)
            returned_path = split_image(test_image, "test", (2, 2), temp_dir, return_tile_dim=(1, 1))
            
            assert returned_path is not None
            assert os.path.exists(returned_path)
            assert "test_1_1.jpg" in returned_path
        
        finally:
            shutil.rmtree(temp_dir)


class TestArgumentParser:
    """Test CLI argument parsing for medical image processing workflows"""
    
    def test_build_argument_parser_structure(self):
        """Test that argument parser has required medical workflow options"""
        # Mock sys.argv to avoid pytest interference
        with patch('sys.argv', ['script.py', '-s', 'source.jpg', '-t', 'target.jpg']):
            args = build_argument_parser()
            
            assert 'source' in args
            assert 'target' in args
            assert 'method' in args
            assert 'color' in args
            assert 'skin_blend' in args
            assert 'hair_blend' in args
            assert 'bg_blend' in args
    
    def test_targeted_transfer_arguments(self):
        """Test targeted transfer specific arguments for medical use"""
        test_args = [
            'script.py', '-s', 'source.jpg', '-t', 'target.jpg',
            '--skin-blend', '0.8', '--hair-blend', '0.6', '--bg-blend', '0.4'
        ]
        
        with patch('sys.argv', test_args):
            args = build_argument_parser()
            
            assert args['skin_blend'] == 0.8
            assert args['hair_blend'] == 0.6
            assert args['bg_blend'] == 0.4
    
    def test_method_choices_validation(self):
        """Test that method choices include all medical algorithms"""
        test_args = ['script.py', '-s', 'source.jpg', '-t', 'target.jpg', '-m', 'targeted']
        
        with patch('sys.argv', test_args):
            args = build_argument_parser()
            assert args['method'] == 'targeted'
        
        # Test other valid methods
        valid_methods = ['kmeans', 'reinhard', 'unique', 'entire', 'targeted', 'all']
        for method in valid_methods:
            test_args = ['script.py', '-s', 'source.jpg', '-t', 'target.jpg', '-m', method]
            with patch('sys.argv', test_args):
                args = build_argument_parser()
                assert args['method'] == method


class TestMedicalComplianceFeatures:
    """Test medical compliance features in helper functions"""
    
    def test_image_stats_precision(self):
        """Test that image statistics have medical-grade precision"""
        # Create known image for testing
        temp_dir = tempfile.mkdtemp()
        test_path = os.path.join(temp_dir, "test.jpg")
        
        try:
            # Create image with known properties
            known_image = np.full((100, 100, 3), [128, 64, 192], dtype=np.uint8)
            cv2.imwrite(test_path, cv2.cvtColor(known_image, cv2.COLOR_RGB2BGR))
            
            stats = get_image_stats(test_path, color_space="RGB")
            height, width, num_pixels, num_unique_colors, ch1_mean, ch1_std, ch2_mean, ch2_std, ch3_mean, ch3_std = stats
            
            assert height == 100
            assert width == 100
            assert num_pixels == 10000
            assert num_unique_colors == 1  # Single color
            
            # Means should be close to expected values (allowing for JPEG compression)
            assert abs(ch1_mean - 128) < 5
            assert abs(ch2_mean - 64) < 5
            assert abs(ch3_mean - 192) < 5
            
            # Standard deviations should be very low for single color
            assert ch1_std < 5
            assert ch2_std < 5
            assert ch3_std < 5
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_color_space_consistency(self):
        """Test consistency across different color spaces for medical analysis"""
        temp_dir = tempfile.mkdtemp()
        test_path = os.path.join(temp_dir, "test.jpg")
        
        try:
            test_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            cv2.imwrite(test_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
            
            # Test different color spaces
            rgb_stats = get_image_stats(test_path, color_space="RGB")
            bgr_stats = get_image_stats(test_path, color_space="BGR")
            lab_stats = get_image_stats(test_path, color_space="LAB")
            
            # Dimensions should be consistent across color spaces
            assert rgb_stats[0] == bgr_stats[0] == lab_stats[0]  # height
            assert rgb_stats[1] == bgr_stats[1] == lab_stats[1]  # width
            assert rgb_stats[2] == bgr_stats[2] == lab_stats[2]  # num_pixels
        
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])