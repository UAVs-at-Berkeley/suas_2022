import cv2
import numpy as np
import os

class VideoStitcher:
    def __init__(self, video_source, config):
        self.video_source = video_source
        self.config = config
        self.frame_buffer = []
        self.stitched_image = None
        self.is_running = False
        self.reference_frame = None
        self.reference_features = None
        self.last_successful_frame = None
        self.last_successful_features = None
        self.frame_count = 0
        self.successful_stitches = 0

    def start_processing(self):
        """
        Main processing loop that handles video capture, frame processing, and stitching.
        """
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                raise ValueError(f"Could not open video source: {self.video_source}")

            self.is_running = True
            self.frame_count = 0

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break

                self.frame_count += 1
                print(f"Processing frame {self.frame_count}")

                # Check if frame is usable (not blurry)
                if not self.is_frame_usable(frame):
                    print(f"Frame {self.frame_count} rejected - too blurry")
                    continue

                # Process the frame
                processed_frame, features = self.process_frame(frame)
                if processed_frame is None:
                    print(f"Frame {self.frame_count} processing failed")
                    continue

                # Update the stitched image
                self.update_stitched_image(processed_frame, features)

                # Optional: Display progress
                if self.frame_count % 10 == 0:
                    print(f"Processed {self.frame_count} frames, {self.successful_stitches} successful stitches")

        except Exception as e:
            print(f"Error in video processing: {str(e)}")
        finally:
            self.is_running = False
            if 'cap' in locals():
                cap.release()
            print("Video processing completed")

    def process_frame(self, frame):
        """
        Process individual frame for stitching.
        
        Args:
            frame: Input frame from video capture
            
        Returns:
            tuple: (processed_frame, features) or None if processing fails
        """
        try:
            # Detect features in the frame
            features = self.detect_features(frame)
            if features is None:
                return None
                
            return frame, features
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None

    def is_frame_usable(self, frame):
        """
        Check if frame is clear enough to use for stitching.
        
        Args:
            frame: Input frame to check
            
        Returns:
            bool: True if frame is usable, False otherwise
        """
        try:
            # Debug mode: accept all frames
            if self.config == "DEBUG":
                return True
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Threshold for blur detection
            # Lower variance means more blur
            blur_threshold = 100.0
            
            if variance < blur_threshold:
                print(f"Frame rejected - too blurry (variance: {variance:.2f})")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error checking frame usability: {str(e)}")
            return False

    def detect_features(self, frame):
        """
        Detect features in a single frame.
        
        Args:
            frame: Input frame to detect features in
            
        Returns:
            dict: Dictionary containing keypoints, descriptors, and grayscale image
                 or None if feature detection fails
        """
        try:
            # Convert to grayscale for feature detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect features using SIFT
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            if keypoints is None or len(keypoints) < 10:
                print("Not enough features detected in frame")
                return None
                
            # Store features for later matching
            features = {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'gray': gray
            }
            
            return features
            
        except Exception as e:
            print(f"Error detecting features: {str(e)}")
            return None

    def update_stitched_image(self, new_frame, new_features):
        """
        Update the final stitched image by incorporating a new frame.
        
        Args:
            new_frame: The new frame to be added to the stitched image
            new_features: Dictionary containing keypoints, descriptors, and grayscale image
            
        Returns:
            None (updates self.stitched_image in place)
        """
        if self.stitched_image is None:
            # First frame - initialize the stitched image and reference frame
            self.stitched_image = new_frame.copy()
            self.reference_frame = new_frame.copy()
            self.reference_features = new_features
            self.last_successful_frame = new_frame.copy()
            self.last_successful_features = new_features
            self.successful_stitches += 1
            return

        try:
            # Try to compute homography with reference frame first
            homography = self.compute_homography(self.reference_features, new_features)
            
            if homography is None:
                # If that fails, try with the last successful frame
                if self.last_successful_frame is not None:
                    homography = self.compute_homography(self.last_successful_features, new_features)
                    
                if homography is None:
                    print("Failed to compute homography with both reference and last successful frame")
                    return

            # Stitch the frames together
            stitched = self.stitch_frames(self.stitched_image, new_frame, homography)
            if stitched is None:
                print("Failed to stitch frames")
                return

            # Verify the stitching quality
            if self.verify_stitching_quality(stitched):
                self.stitched_image = stitched
                self.last_successful_frame = new_frame.copy()
                self.last_successful_features = new_features
                self.successful_stitches += 1
            else:
                print("Stitching quality check failed")

        except Exception as e:
            print(f"Error updating stitched image: {str(e)}")
            return

    def verify_stitching_quality(self, stitched_image):
        """
        Verify the quality of the stitched image.
        
        Args:
            stitched_image: The stitched image to verify
            
        Returns:
            bool: True if the stitching quality is acceptable, False otherwise
        """
        try:
            # Check if the image is too large (indicating potential error)
            h, w = stitched_image.shape[:2]
            if w > 10000 or h > 10000:
                print("Stitched image too large - likely incorrect transformation")
                return False

            # Check if the image has reasonable dimensions
            if w < 100 or h < 100:
                print("Stitched image too small - likely incorrect transformation")
                return False

            # Check if the image has reasonable content
            if np.mean(stitched_image) < 10 or np.mean(stitched_image) > 250:
                print("Stitched image has unreasonable pixel values")
                return False

            return True

        except Exception as e:
            print(f"Error verifying stitching quality: {str(e)}")
            return False

    def compute_homography(self, features1, features2):
        """
        Compute transformation between two frames using their features.
        
        Args:
            features1: Dictionary containing keypoints, descriptors, and grayscale image of first frame
            features2: Dictionary containing keypoints, descriptors, and grayscale image of second frame
            
        Returns:
            numpy.ndarray: 3x3 homography matrix or None if computation fails
        """
        try:
            # Create feature matcher
            bf = cv2.BFMatcher()
            
            # Match features between the two frames
            matches = bf.knnMatch(features1['descriptors'], features2['descriptors'], k=2)
            
            # Apply ratio test to get good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) < 10:
                print("Not enough good matches found for homography")
                return None
            
            # Get corresponding points
            src_pts = np.float32([features1['keypoints'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([features2['keypoints'][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography using RANSAC
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                print("Failed to compute homography")
                return None

            # Verify the homography is reasonable
            if not self.verify_homography(H, features1['gray'].shape, features2['gray'].shape):
                print("Homography verification failed")
                return None
                
            return H
            
        except Exception as e:
            print(f"Error computing homography: {str(e)}")
            return None

    def verify_homography(self, H, shape1, shape2):
        """
        Verify if the computed homography is reasonable.
        
        Args:
            H: 3x3 homography matrix
            shape1: Shape of the first image (height, width)
            shape2: Shape of the second image (height, width)
            
        Returns:
            bool: True if the homography is reasonable, False otherwise
        """
        try:
            # Check determinant
            det = np.linalg.det(H)
            if abs(det) < 0.1 or abs(det) > 10:
                print(f"Unreasonable homography determinant: {det}")
                return False

            # Check if the transformation is too extreme
            h1, w1 = shape1
            h2, w2 = shape2
            
            # Transform corners of second image
            corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners2, H)
            
            # Check if the transformed corners are within reasonable bounds
            x_min, y_min = warped_corners[:, 0, 0].min(), warped_corners[:, 0, 1].min()
            x_max, y_max = warped_corners[:, 0, 0].max(), warped_corners[:, 0, 1].max()
            
            # Calculate the scale of the transformation
            scale_x = (x_max - x_min) / w2
            scale_y = (y_max - y_min) / h2
            
            if scale_x > 3 or scale_y > 3 or scale_x < 0.33 or scale_y < 0.33:
                print(f"Unreasonable transformation scale: x={scale_x}, y={scale_y}")
                return False

            return True

        except Exception as e:
            print(f"Error verifying homography: {str(e)}")
            return False

    def stitch_frames(self, frame1, frame2, homography):
        """
        Stitch two frames together using the computed homography.
        
        Args:
            frame1: First frame (existing stitched image)
            frame2: Second frame (new frame to add)
            homography: 3x3 homography matrix
            
        Returns:
            numpy.ndarray: Stitched image or None if stitching fails
        """
        try:
            # Validate inputs
            if frame1 is None or frame2 is None or homography is None:
                print("Invalid input: frames or homography is None")
                return None
                
            if frame1.size == 0 or frame2.size == 0:
                print("Invalid input: empty frames")
                return None
                
            # Validate homography matrix
            if not isinstance(homography, np.ndarray) or homography.shape != (3, 3):
                print("Invalid homography matrix")
                return None
                
            # Check for degenerate homography
            if abs(np.linalg.det(homography)) < 1e-6:
                print("Degenerate homography matrix")
                return None
            
            # Get original dimensions
            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            
            # Calculate target size based on the larger frame
            target_height = max(h1, h2)
            target_width = max(w1, w2)
            
            # Resize frames to target size
            if frame1.shape[:2] != (target_height, target_width):
                frame1 = cv2.resize(frame1, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            if frame2.shape[:2] != (target_height, target_width):
                frame2 = cv2.resize(frame2, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Update dimensions after resizing
            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            
            # Get the corners of the second image
            corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            
            # Transform the corners using the homography
            warped_corners = cv2.perspectiveTransform(corners2, homography)
            
            # Find the dimensions of the new canvas with padding
            padding = 10  # Add padding to prevent edge artifacts
            x_min = min(0, warped_corners[:, 0, 0].min()) - padding
            x_max = max(w1, warped_corners[:, 0, 0].max()) + padding
            y_min = min(0, warped_corners[:, 0, 1].min()) - padding
            y_max = max(h1, warped_corners[:, 0, 1].max()) + padding
            
            # Calculate the new canvas size
            new_width = int(np.ceil(x_max - x_min))
            new_height = int(np.ceil(y_max - y_min))
            
            # Check if canvas is too large and downsample if necessary
            max_dimension = 10000  # Maximum dimension in pixels
            scale = 1.0
            if new_width > max_dimension or new_height > max_dimension:
                scale = min(max_dimension / new_width, max_dimension / new_height)
                new_width = int(new_width * scale)
                new_height = int(new_height * scale)
                # Scale the homography
                scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
                homography = scale_matrix.dot(homography)
            
            # Calculate the translation needed to make all coordinates positive
            translation = np.array([
                [1, 0, -x_min],
                [0, 1, -y_min],
                [0, 0, 1]
            ])
            
            # Adjust the homography to account for the translation
            adjusted_homography = translation.dot(homography)
            
            # Warp the second image using the inverse of the adjusted homography
            warped = cv2.warpPerspective(frame2, np.linalg.inv(adjusted_homography), (new_width, new_height))
            
            # Create a mask for the warped image
            mask = np.zeros((new_height, new_width), dtype=np.uint8)
            adjusted_corners = warped_corners + [-x_min, -y_min]
            
            # Scale corners if we downsampled
            if scale != 1.0:
                adjusted_corners *= scale
            
            # Ensure corners are within image bounds
            adjusted_corners = np.clip(adjusted_corners, 0, [new_width-1, new_height-1])
            
            # Create a convex hull of the corners to ensure convexity
            hull = cv2.convexHull(adjusted_corners.astype(np.int32))
            mask = cv2.fillConvexPoly(mask, hull, 255)
            
            # Dilate the mask to ensure proper blending
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Create the new canvas with the first image
            canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            
            # Calculate the region where the first image should be placed
            x_start = max(0, int(-x_min * scale))
            y_start = max(0, int(-y_min * scale))
            x_end = min(new_width, int((w1 - x_min) * scale))
            y_end = min(new_height, int((h1 - y_min) * scale))
            
            # Place the first image on the canvas
            if x_end > x_start and y_end > y_start:
                # Calculate source region
                src_x_start = max(0, int(-x_min))
                src_y_start = max(0, int(-y_min))
                src_x_end = min(w1, int(new_width/scale - x_min))
                src_y_end = min(h1, int(new_height/scale - y_min))
                
                # Ensure all indices are integers and within bounds
                src_x_start, src_y_start = int(src_x_start), int(src_y_start)
                src_x_end, src_y_end = int(src_x_end), int(src_y_end)
                x_start, y_start = int(x_start), int(y_start)
                x_end, y_end = int(x_end), int(y_end)
                
                # Verify all indices are within bounds
                if (src_x_end > src_x_start and src_y_end > src_y_start and
                    x_end > x_start and y_end > y_start and
                    src_x_end <= w1 and src_y_end <= h1 and
                    x_end <= new_width and y_end <= new_height):
                    
                    try:
                        # Resize the source region if necessary
                        if (src_x_end - src_x_start) != (x_end - x_start) or \
                           (src_y_end - src_y_start) != (y_end - y_start):
                            source_region = frame1[src_y_start:src_y_end, src_x_start:src_x_end]
                            source_region = cv2.resize(source_region, (x_end - x_start, y_end - y_start))
                            canvas[y_start:y_end, x_start:x_end] = source_region
                        else:
                            canvas[y_start:y_end, x_start:x_end] = frame1[src_y_start:src_y_end, src_x_start:src_x_end]
                    except ValueError as e:
                        print(f"Error placing first image: {str(e)}")
                        return None
            
            # Calculate the center of the overlapping region for blending
            overlap_center = np.mean(hull, axis=0).astype(int)
            center = (overlap_center[0][0], overlap_center[0][1])
            
            # Ensure center is within the image bounds and the mask
            center = (max(0, min(new_width-1, center[0])), 
                     max(0, min(new_height-1, center[1])))
            
            # Verify center is within the mask
            if mask[center[1], center[0]] == 0:
                # Find the closest point in the mask
                mask_points = np.where(mask > 0)
                if len(mask_points[0]) > 0:
                    distances = (mask_points[0] - center[1])**2 + (mask_points[1] - center[0])**2
                    closest_idx = np.argmin(distances)
                    center = (mask_points[1][closest_idx], mask_points[0][closest_idx])
                else:
                    print("No valid blending center found")
                    return None
            
            # Check if there's enough overlap for blending
            overlap_area = np.sum(mask) / 255
            if overlap_area < 100:  # Minimum overlap area in pixels
                print("Insufficient overlap between frames")
                return None
            
            # Blend the images
            stitched = cv2.seamlessClone(warped, canvas, mask, center, cv2.NORMAL_CLONE)
            
            return stitched
            
        except Exception as e:
            print(f"Error stitching frames: {str(e)}")
            return None

    def save_result(self, output_path):
        """
        Save the final stitched image to the specified path.
        
        Args:
            output_path: Path where the stitched image should be saved
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Validate inputs
            if self.stitched_image is None:
                print("No stitched image to save")
                return False
                
            if not output_path:
                print("No output path specified")
                return False
                
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save the image
            success = cv2.imwrite(output_path, self.stitched_image)
            
            if not success:
                print(f"Failed to save image to {output_path}")
                return False
                
            print(f"Successfully saved stitched image to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving stitched image: {str(e)}")
            return False
    
if __name__ == "__main__":
    # Initialize video stitcher in debug mode
    video_stitcher = VideoStitcher("C:/Users/isaac/Downloads/extrashort.mp4", "DEBUG")
    
    print("Starting video processing...")
    video_stitcher.start_processing()
    
    print("Saving result...")
    success = video_stitcher.save_result("C:/Users/isaac/Downloads/stitched_image.jpg")
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")
