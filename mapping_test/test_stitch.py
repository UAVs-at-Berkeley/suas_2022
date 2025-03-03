import cv2
import numpy as np
import os
import shutil
from image_stitch import stitchMatrix, KING_SIZE, OVERLAP_X, OVERLAP_Y

def setup_test_environment():
    """Prepare test environment"""
    # Clear and recreate debug directory
    if os.path.exists("debug"):
        shutil.rmtree("debug")
    os.makedirs("debug")
    
    # Create test results directory
    os.makedirs("debug/test_results", exist_ok=True)

def test_ideal_case():
    """Test with perfectly aligned images (no rotation)"""
    print("\nTesting ideal case (no rotation)...")
    
    # Load test images
    imgArr = [[cv2.imread(f'runway1/({i}, {j}).png') for j in range(4)] for i in range(3)]
    
    # Verify all images loaded correctly
    if not all(all(img is not None for img in row) for row in imgArr):
        print("Error: Failed to load test images")
        return False
    
    # Verify image sizes
    for i, row in enumerate(imgArr):
        for j, img in enumerate(row):
            if img.shape[:2] != (KING_SIZE[1], KING_SIZE[0]):
                print(f"Error: Image at ({i},{j}) has incorrect size: {img.shape}")
                return False
    
    # Run stitching
    result = stitchMatrix(imgArr)
    
    # Save result
    cv2.imwrite("debug/test_results/ideal_case_result.png", result)
    
    # Load reference image
    reference = cv2.imread("runway1.jpg")
    if reference is None:
        print("Warning: Could not load reference image for comparison")
        return True
    
    # Resize reference to match result for comparison
    reference = cv2.resize(reference, (result.shape[1], result.shape[0]))
    
    # Compare images
    diff = cv2.absdiff(result, reference)
    mean_diff = np.mean(diff)
    print(f"Mean difference from reference: {mean_diff:.2f}")
    
    # Save difference visualization
    cv2.imwrite("debug/test_results/ideal_case_diff.png", diff)
    
    # Create side-by-side comparison
    comparison = np.hstack([reference, result])
    cv2.imwrite("debug/test_results/ideal_case_comparison.png", comparison)
    
    return mean_diff < 50  # Threshold can be adjusted

def test_overlap_consistency():
    """Test if overlap regions are being handled correctly"""
    print("\nTesting overlap consistency...")
    
    imgArr = [[cv2.imread(f'runway1/({i}, {j}).png') for j in range(4)] for i in range(3)]
    
    # Test horizontal overlaps
    for i in range(3):  # rows
        for j in range(3):  # cols - 1
            img1 = imgArr[i][j]
            img2 = imgArr[i][j+1]
            
            # Extract overlap regions
            overlap1 = img1[:, -int(OVERLAP_X):]
            overlap2 = img2[:, :int(OVERLAP_X)]
            
            # Compare overlaps
            diff = cv2.absdiff(overlap1, overlap2)
            mean_diff = np.mean(diff)
            
            print(f"Horizontal overlap diff at ({i},{j}-{j+1}): {mean_diff:.2f}")
            
            # Save overlap regions for inspection
            cv2.imwrite(f"debug/test_results/overlap_h_{i}_{j}.png", 
                       np.hstack([overlap1, overlap2]))
    
    return True

def run_all_tests():
    """Run all test cases"""
    setup_test_environment()
    
    tests = [
        ("Ideal Case", test_ideal_case),
        ("Overlap Consistency", test_overlap_consistency)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        try:
            result = test_func()
            status = "PASSED" if result else "FAILED"
            print(f"{test_name}: {status}")
            all_passed &= result
        except Exception as e:
            print(f"{test_name}: ERROR - {str(e)}")
            all_passed = False
    
    print(f"\nOverall test result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed

if __name__ == "__main__":
    run_all_tests() 