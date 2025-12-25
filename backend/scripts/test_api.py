import requests
import base64
import cv2
import time
import os

API_URL = "http://localhost:8000/api/v1"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_frame_processing(image_path="test_image.jpg"):
    """Test single frame processing"""
    print(f"\nTesting frame processing with {image_path}...")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Test image not found: {image_path}")
        print("Creating a test image...")
        # Create a simple test image
        test_image = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (200, 200), (255, 255, 255), -1)
        cv2.imwrite(image_path, test_image)
    
    # Read and encode image
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Prepare request
    request_data = {
        "image_data": encoded_image,
        "return_format": "base64"
    }
    
    # Send request
    start_time = time.time()
    response = requests.post(f"{API_URL}/process/frame", json=request_data)
    processing_time = (time.time() - start_time) * 1000
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"API Processing time: {result['processing_time_ms']:.1f}ms")
        print(f"Total time (including network): {processing_time:.1f}ms")
        
        # Save result
        if result.get('output_data'):
            output_data = base64.b64decode(result['output_data'])
            with open("processed_frame.jpg", "wb") as f:
                f.write(output_data)
            print("Saved processed frame as 'processed_frame.jpg'")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_video_upload(video_path="test_video.mp4"):
    """Test video upload and processing"""
    print(f"\nTesting video upload with {video_path}...")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Test video not found: {video_path}")
        return False
    
    # Upload video
    with open(video_path, "rb") as video_file:
        files = {"file": (os.path.basename(video_path), video_file, "video/mp4")}
        response = requests.post(f"{API_URL}/process/video/upload", files=files)
    
    print(f"Upload status: {response.status_code}")
    
    if response.status_code == 202:
        result = response.json()
        processing_id = result['processing_id']
        status_url = result['status_url']
        
        print(f"Processing ID: {processing_id}")
        print(f"Status URL: {status_url}")
        
        # Poll for status
        print("Polling for status...")
        for i in range(30):  # Max 30 checks
            time.sleep(2)
            status_response = requests.get(f"{API_URL}/process/status/{processing_id}")
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data['status']
                progress = status_data['progress']
                
                print(f"  Progress: {progress*100:.1f}% - Status: {status}")
                
                if status == "completed":
                    print("\n✅ Video processing completed!")
                    
                    # Download the result
                    download_url = f"{API_URL}/process/download/{processing_id}"
                    download_response = requests.get(download_url, stream=True)
                    
                    if download_response.status_code == 200:
                        output_path = f"processed_{os.path.basename(video_path)}"
                        with open(output_path, "wb") as f:
                            for chunk in download_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        print(f"Downloaded processed video: {output_path}")
                        return True
                
                elif status == "failed":
                    print(f"\n❌ Processing failed: {status_data.get('error')}")
                    return False
            else:
                print(f"  Error checking status: {status_response.status_code}")
        
        print("\n⚠ Timeout waiting for processing")
        return False
    else:
        print(f"Upload failed: {response.text}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting model info endpoint...")
    response = requests.get(f"{API_URL}/model/info")
    
    if response.status_code == 200:
        model_info = response.json()
        print(f"Model: {model_info['name']}")
        print(f"Parameters: {model_info['parameters']:,}")
        print(f"Device: {model_info['device']}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def create_test_video():
    """Create a simple test video if none exists"""
    print("\nCreating test video...")
    
    # Create a 5-second test video
    output_path = "test_video.mp4"
    fps = 10
    duration = 5
    total_frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (256, 256))
    
    for i in range(total_frames):
        # Create a simple animated frame
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Animate a moving circle
        center_x = 128 + int(100 * np.sin(2 * np.pi * i / total_frames))
        center_y = 128 + int(100 * np.cos(2 * np.pi * i / total_frames))
        
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {output_path} ({total_frames} frames)")
    return output_path

def main():
    """Run all tests"""
    print("="*60)
    print("Testing Pix2Pix Video API")
    print("="*60)
    
    # Wait for API to start
    print("Waiting for API to be ready...")
    for i in range(30):
        try:
            if test_health():
                print("✅ API is ready!")
                break
        except requests.exceptions.ConnectionError:
            print(f"  Attempt {i+1}/30: API not ready yet...")
            time.sleep(1)
    else:
        print("❌ API failed to start")
        return
    
    # Create test video if needed
    if not os.path.exists("test_video.mp4"):
        create_test_video()
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Frame Processing", lambda: test_frame_processing("test_image.jpg")),
        ("Video Processing", lambda: test_video_upload("test_video.mp4"))
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print('='*40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
        except Exception as e:
            print(f"Error: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {total - passed} tests failed")

if __name__ == "__main__":
    import numpy as np
    main()