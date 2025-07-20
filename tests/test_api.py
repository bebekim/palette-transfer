# ABOUTME: Test suite for Flask API endpoints for hair clinic palette transfer SaaS
# ABOUTME: Tests health endpoint, palette transfer API, and medical image validation

import json
from datetime import datetime

# Import our Flask app
from app import app


class TestHealthEndpoint:
    """Test the health check endpoint"""
    
    def test_health_endpoint_returns_200(self):
        """Test that health endpoint returns 200 status"""
        with app.test_client() as client:
            response = client.get('/api/v1/health')
            assert response.status_code == 200
    
    def test_health_endpoint_returns_json(self):
        """Test that health endpoint returns valid JSON"""
        with app.test_client() as client:
            response = client.get('/api/v1/health')
            data = json.loads(response.data)
            assert isinstance(data, dict)
    
    def test_health_endpoint_contains_required_fields(self):
        """Test that health endpoint returns required status fields"""
        with app.test_client() as client:
            response = client.get('/api/v1/health')
            data = json.loads(response.data)
            
            assert 'status' in data
            assert 'timestamp' in data
            assert 'service' in data
            assert data['status'] == 'healthy'
            assert data['service'] == 'palette-transfer-api'
    
    def test_health_endpoint_timestamp_format(self):
        """Test that health endpoint returns valid ISO timestamp"""
        with app.test_client() as client:
            response = client.get('/api/v1/health')
            data = json.loads(response.data)
            
            # Should be able to parse the timestamp
            timestamp = datetime.fromisoformat(data['timestamp'])
            assert isinstance(timestamp, datetime)


class TestPaletteTransferEndpoint:
    """Test the main palette transfer endpoint"""
    
    def test_palette_transfer_endpoint_exists(self):
        """Test that palette transfer endpoint responds to POST"""
        with app.test_client() as client:
            response = client.post('/api/v1/palette-transfer')
            # Should not be 404 (endpoint exists)
            assert response.status_code != 404
    
    def test_palette_transfer_requires_json(self):
        """Test that endpoint requires JSON content type"""
        with app.test_client() as client:
            response = client.post('/api/v1/palette-transfer', 
                                 data='not json',
                                 content_type='text/plain')
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert 'json' in data['error'].lower()
    
    def test_palette_transfer_requires_source_and_target(self):
        """Test that endpoint requires both source_image and target_image"""
        with app.test_client() as client:
            # Missing both
            response = client.post('/api/v1/palette-transfer',
                                 json={})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert 'source_image' in data['error'] and 'target_image' in data['error']
            
            # Missing target_image
            response = client.post('/api/v1/palette-transfer',
                                 json={'source_image': 'fake_base64'})
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert 'target_image' in data['error']


class TestImageValidation:
    """Test image validation - basic vs medical mode"""
    
    def test_basic_validation_accepts_small_images(self):
        """Test that basic validation accepts smaller images (non-medical use)"""
        # Create a simple 1x1 pixel base64 image
        small_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        with app.test_client() as client:
            response = client.post('/api/v1/palette-transfer',
                                 json={
                                     'source_image': small_image_b64,
                                     'target_image': small_image_b64,
                                     'medical_mode': False  # Basic validation
                                 })
            # Should accept small images in basic mode
            assert response.status_code != 400 or 'too small' not in response.get_json().get('error', '').lower()
    
    def test_medical_validation_rejects_small_images(self):
        """Test that medical validation rejects images too small for medical documentation"""
        # Create a simple 1x1 pixel base64 image  
        small_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        with app.test_client() as client:
            response = client.post('/api/v1/palette-transfer',
                                 json={
                                     'source_image': small_image_b64,
                                     'target_image': small_image_b64,
                                     'medical_mode': True  # Strict medical validation
                                 })
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert 'medical documentation' in data['error'].lower()
    
    def test_medical_mode_defaults_to_false(self):
        """Test that medical_mode defaults to False when not specified"""
        small_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        with app.test_client() as client:
            response = client.post('/api/v1/palette-transfer',
                                 json={
                                     'source_image': small_image_b64,
                                     'target_image': small_image_b64
                                     # medical_mode not specified - should default to False
                                 })
            # Should use basic validation (accept small images)
            assert response.status_code != 400 or 'medical documentation' not in response.get_json().get('error', '').lower()
    
    def test_invalid_image_format_rejected_in_all_modes(self):
        """Test that invalid image data is rejected in both basic and medical modes"""
        invalid_image = "not_valid_base64_image_data"
        
        with app.test_client() as client:
            # Test basic mode
            response = client.post('/api/v1/palette-transfer',
                                 json={
                                     'source_image': invalid_image,
                                     'target_image': invalid_image,
                                     'medical_mode': False
                                 })
            assert response.status_code == 400
            
            # Test medical mode
            response = client.post('/api/v1/palette-transfer',
                                 json={
                                     'source_image': invalid_image,
                                     'target_image': invalid_image,
                                     'medical_mode': True
                                 })
            assert response.status_code == 400