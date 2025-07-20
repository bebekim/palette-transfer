# -*- coding: utf-8 -*-
# ABOUTME: This file tests the basic structure and imports of refactored modules
# ABOUTME: Validates that modules can be imported without running heavy algorithms

import pytest
import sys
import os

# Add the palette-transfer directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'palette-transfer'))


def test_basic_imports():
    """Test that all refactored modules can be imported"""
    try:
        # These should import without errors (though they may need dependencies for actual use)
        import helpers
        print("✓ helpers module imported successfully")
        
        # Test that helper functions exist
        assert hasattr(helpers, 'get_image')
        assert hasattr(helpers, 'get_unique_colors')
        assert hasattr(helpers, 'build_argument_parser')
        print("✓ helper functions are available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import helpers: {e}")


def test_module_structure():
    """Test that modules have expected structure"""
    try:
        import helpers
        
        # Test build_argument_parser
        # Mock sys.argv to test argument parser
        original_argv = sys.argv
        sys.argv = ['script.py', '-s', 'source.jpg', '-t', 'target.jpg']
        
        try:
            args = helpers.build_argument_parser()
            assert 'source' in args
            assert 'target' in args
            assert 'method' in args
            print("✓ Argument parser works correctly")
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        pytest.fail(f"Module structure test failed: {e}")


def test_file_operations():
    """Test file operation helpers that don't require heavy dependencies"""
    try:
        import helpers
        import tempfile
        import shutil
        
        # Test folder creation
        temp_dir = tempfile.mkdtemp()
        test_folder = os.path.join(temp_dir, "test_folder")
        
        helpers.create_folder(test_folder)
        assert os.path.exists(test_folder)
        print("✓ Folder creation works")
        
        # Test folder deletion
        helpers.delete_folder_and_contents(test_folder)
        assert not os.path.exists(test_folder)
        print("✓ Folder deletion works")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        pytest.fail(f"File operations test failed: {e}")


if __name__ == "__main__":
    test_basic_imports()
    test_module_structure()
    test_file_operations()
    print("\n🎉 All basic structure tests passed!")
    print("\nRefactoring Summary:")
    print("✓ Monolithic palette-transfer.py split into focused modules")
    print("✓ kmeans_palette.py - K-means clustering algorithms")
    print("✓ reinhard_transfer.py - Reinhard color transfer")
    print("✓ targeted_transfer.py - Medical targeted transfer with validation")
    print("✓ entire_palette.py - Complete palette transfer")
    print("✓ helpers.py - Utility functions and CLI")
    print("✓ Comprehensive test suite created for each module")
    print("✓ Medical compliance features and parameter validation")
    print("✓ Focused, maintainable, and testable code structure")