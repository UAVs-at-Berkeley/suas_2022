import cv2
import numpy as np
import os

class Config:
    def __init__(self, debug=False, maxframes=-1, padding_multiplier=0.2, debug_save_dir=None):
        self.debug = debug
        self.maxframes = maxframes
        self.padding_multiplier = padding_multiplier
        if debug_save_dir is None:
            self.debug_save_dir = os.path.join(os.path.dirname(__file__), "stitch_debug")
        else:
            self.debug_save_dir = debug_save_dir

class VideoStitcher:
    def __init__(self):
        self.video_source = None
        self.config = None
        self.stitched_image = None
        self.is_running = False
        self.frame_count = 0
        self.successful_stitches = 0
        self.debug_save_dir = None  # Directory for debug images
        
    def configure(self, *args):
        self.config = Config(*args)
        self.max_frames = self.config.maxframes
        self.padding_multiplier = self.config.padding_multiplier
        self.debug_save_dir = self.config.debug_save_dir

    def pad_frame(self, frame, padding_multiplier=0.2):
        """
        Pad a frame dynamically based on content proximity to edges.
        If content is within 20% of any edge, expands padding in that direction.
        
        Args:
            frame: Input frame to pad
            padding_multiplier: Base multiplier for padding size (default: 1.0)
            
        Returns:
            numpy.ndarray: Padded frame
        """
        try:
            h, w = frame.shape[:2]
            
            # Convert to grayscale and find non-zero pixels
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            non_zero = cv2.findNonZero(gray)
            
            if non_zero is None:
                # If no non-zero pixels found, use default padding
                padded_h = h + int(2 * h * padding_multiplier)
                padded_w = w + int(2 * w * padding_multiplier)
                y_offset = int(h * padding_multiplier)
                x_offset = int(w * padding_multiplier)
            else:
                # Get bounding box of non-zero pixels
                x, y, w_content, h_content = cv2.boundingRect(non_zero)
                
                # Calculate distances to edges
                dist_left = x
                dist_right = w - (x + w_content)
                dist_top = y
                dist_bottom = h - (y + h_content)
                
                # Calculate required padding for each edge
                pad_left = int(w * padding_multiplier) if dist_left < w * padding_multiplier else 0
                pad_right = int(w * padding_multiplier) if dist_right < w * padding_multiplier else 0
                pad_top = int(h * padding_multiplier) if dist_top < h * padding_multiplier else 0
                pad_bottom = int(h * padding_multiplier) if dist_bottom < h * padding_multiplier else 0
                
                # Calculate final dimensions and offsets
                padded_h = h + pad_top + pad_bottom
                padded_w = w + pad_left + pad_right
                y_offset = pad_top
                x_offset = pad_left
            
            # Create padded image with black background
            padded_image = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
            
            # Place the frame in the calculated position
            padded_image[y_offset:y_offset + h, x_offset:x_offset + w] = frame
            
            return padded_image
            
        except Exception as e:
            print(f"Error padding frame: {str(e)}")
            return frame

    def start_processing(self, video_source):
        """
        Main processing loop that handles video capture, frame processing, and stitching.
        """
        try:
            self.video_source = video_source
            # Initialize video capture
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                raise ValueError(f"Could not open video source: {self.video_source}")

            self.is_running = True
            self.frame_count = 0

            while self.is_running and self.frame_count < self.max_frames:
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
            if self.config.debug == True:
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

    def save_debug_image(self, image, frame_num, stitch_num, description=""):
        """
        Save a debug image with frame and stitch information.
        
        Args:
            image: The image to save
            frame_num: Current frame number
            stitch_num: Current stitch number
            description: Additional description for the filename
        """
        try:
            # Create debug directory if it doesn't exist
            if not os.path.exists(self.debug_save_dir):
                os.makedirs(self.debug_save_dir)
            
            # Create filename with frame and stitch information
            filename = f"stitch_f{frame_num:03d}_s{stitch_num:03d}"
            if description:
                filename += f"_{description}"
            filename += ".jpg"
            
            # Save the image
            filepath = os.path.join(self.debug_save_dir, filename)
            cv2.imwrite(filepath, image)
            print(f"Saved debug image: {filename}")
            
        except Exception as e:
            print(f"Error saving debug image: {str(e)}")

    def update_stitched_image(self, new_frame, new_features):
        """
        Update the final stitched image by incorporating a new frame.
        
        Args:
            new_frame: The new frame to be added to the stitched image
            new_features: Dictionary containing keypoints, descriptors, and grayscale image
            
        Returns:
            None (updates self.stitched_image in place)
        """
        try:
            # If this is the first frame, initialize the stitched image
            if self.stitched_image is None:
                self.stitched_image = self.pad_frame(new_frame.copy(), 0.2)
                self.reference_frame = new_frame
                self.reference_features = new_features
                self.successful_stitches += 1
                return

            # Detect features in the stitched image
            stitched_features = self.detect_features(self.stitched_image)
            if stitched_features is None:
                print("Failed to detect features in stitched image")
                return

            # Compute homography between new frame and stitched image
            H = self.compute_homography(stitched_features, new_features)
            if H is None:
                print("Failed to compute homography for frame stitching")
                return

            # Warp the new frame using the homography
            h, w = self.stitched_image.shape[:2]
            warped = cv2.warpPerspective(new_frame, H, (w, h))

            # Create binary masks for valid pixels
            stitched_valid = cv2.cvtColor(self.stitched_image, cv2.COLOR_BGR2GRAY) > 0
            warped_valid = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0

            # Create combined mask for overlapping regions
            overlap_mask = stitched_valid & warped_valid

            # For non-overlapping regions, use the original masks
            stitched_only = stitched_valid & ~warped_valid
            warped_only = warped_valid & ~stitched_valid

            # Initialize the result with the stitched image
            result = self.stitched_image.copy()

            # For overlapping regions, take the maximum value
            result[overlap_mask] = np.maximum(self.stitched_image[overlap_mask], warped[overlap_mask])

            # For warped-only regions, use the warped image
            result[warped_only] = warped[warped_only]

            self.stitched_image = result
            self.stitched_image = self.pad_frame(self.stitched_image, 0.2)

            # Update reference frame and features
            self.reference_frame = new_frame
            self.reference_features = new_features
            self.successful_stitches += 1

            # Save debug image if in DEBUG mode
            if self.config.debug == True:
                self.save_debug_image(new_frame, self.frame_count, self.successful_stitches, "new_frame")
                self.save_debug_image(self.stitched_image, self.frame_count, self.successful_stitches, "stitched")

        except Exception as e:
            print(f"Error updating stitched image: {str(e)}")

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

            # Debug visualization if in DEBUG mode
            if self.config.debug == True:
                self.visualize_homography_matches(features1, features2, good_matches, H)
                
            return H
            
        except Exception as e:
            print(f"Error computing homography: {str(e)}")
            return None

    def visualize_homography_matches(self, features1, features2, good_matches, H):
        """
        Create and save a visualization of matched features and homography matrix.
        Shows the images overlaid on top of each other using the homography transformation.
        
        Args:
            features1: Dictionary containing keypoints and grayscale image of first frame
            features2: Dictionary containing keypoints and grayscale image of second frame
            good_matches: List of good feature matches between the frames
            H: 3x3 homography matrix
        """
        try:
            # Convert grayscale images to BGR for visualization
            img1 = cv2.cvtColor(features1['gray'], cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(features2['gray'], cv2.COLOR_GRAY2BGR)
            
            # Get dimensions of first image
            h, w = img1.shape[:2]
            
            # Warp the second image using the homography
            warped_img2 = cv2.warpPerspective(img2, H, (w, h))
            
            # Create an overlay by blending the images
            # Use a simple alpha blending with 0.5 weight for each image
            overlay = cv2.addWeighted(img1, 0.5, warped_img2, 0.5, 0)
            
            # Draw matches on the overlay
            for match in good_matches:
                pt1 = features1['keypoints'][match.queryIdx].pt
                pt2 = features2['keypoints'][match.trainIdx].pt
                
                # Transform pt2 using the homography
                pt2_homogeneous = np.array([pt2[0], pt2[1], 1])
                pt2_transformed = H @ pt2_homogeneous
                pt2_transformed = pt2_transformed[:2] / pt2_transformed[2]
                
                # Draw points and lines
                cv2.circle(overlay, (int(pt1[0]), int(pt1[1])), 3, (0, 255, 0), -1)
                cv2.circle(overlay, (int(pt2_transformed[0]), int(pt2_transformed[1])), 3, (0, 255, 0), -1)
                cv2.line(overlay, (int(pt1[0]), int(pt1[1])), 
                        (int(pt2_transformed[0]), int(pt2_transformed[1])), (0, 255, 0), 1)
            
            # Add homography matrix as text
            homography_text = "Homography Matrix:\n"
            for row in H:
                homography_text += " ".join([f"{x:.3f}" for x in row]) + "\n"
            
            # Add text to image
            y_offset = 30
            for line in homography_text.split('\n'):
                cv2.putText(overlay, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            # Save debug image
            self.save_debug_image(overlay, self.frame_count, self.successful_stitches, "homography_matches")
            
        except Exception as e:
            print(f"Error creating homography visualization: {str(e)}")

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
            
            # More strict scale limits
            if scale_x > 2 or scale_y > 2 or scale_x < 0.5 or scale_y < 0.5:
                print(f"Unreasonable transformation scale: x={scale_x}, y={scale_y}")
                return False

            return True

        except Exception as e:
            print(f"Error verifying homography: {str(e)}")
            return False

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
    video_stitcher = VideoStitcher()
    video_stitcher.configure(True, float('inf'), 0.2, "C:/Users/isaac/Downloads/stitch_debug")
    
    print("Starting video processing...")
    video_stitcher.start_processing("C:/Users/isaac/Downloads/extrashort.mp4")
    
    print("Saving result...")
    success = video_stitcher.save_result("C:/Users/isaac/Downloads/stitched_image.jpg")
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")
