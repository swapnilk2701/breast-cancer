"""
Pectoral Muscle Removal Module for Mammography Preprocessing.

This module detects and removes the pectoral muscle from Mediolateral Oblique (MLO)
views in mammograms. The pectoral muscle appears as a bright, high-intensity triangular
structure located in either the top-left or top-right corner of MLO mammography crops.

Removing the pectoral muscle is a critical preprocessing step in AI/CAD mammography pipelines because:
1. Pectoral muscle tissue has pixel intensity characteristics similar to dense fibroglandular
   breast parenchyma and radiopaque mass lesions.
2. Including the pectoral muscle can cause false-positive detections in deep learning segmentation/classification.
3. Excluding non-breast tissue focuses feature extraction exclusively on clinically relevant mammary tissue.

Steps:
1. Breast Laterality & Orientation Detection: Determine if pectoral muscle lies in the top-left or top-right corner.
2. Quadrant ROI Extraction & Adaptive Otsu Thresholding: Binarize high-intensity candidate muscle region.
3. Canny Edge Detection & Probabilistic Hough Line Fitting: Identify the diagonal boundary separating muscle from breast tissue.
4. Convex Polygon Masking & Morphological Smoothing: Generate binary mask and set pectoral muscle pixels to zero background.
"""

from typing import Tuple, Union, Optional, List
import cv2
import numpy as np
import matplotlib.pyplot as plt


class PectoralMuscleRemover:
    """
    Production-ready Pectoral Muscle Detector and Remover for Mammograms.
    
    Provides automated orientation detection, boundary edge detection, polygon masking,
    and batch execution for single mammography images or large image batches.
    """

    def __init__(
        self,
        roi_fraction_h: float = 0.5,
        roi_fraction_w: float = 0.5,
        canny_thresh1: int = 30,
        canny_thresh2: int = 100,
        fill_value: int = 0
    ) -> None:
        """
        Initialize PectoralMuscleRemover with configurable detection parameters.

        Args:
            roi_fraction_h (float): Fraction of image height (from top) to search for pectoral muscle (default: 0.5).
                For MLO views, the muscle typically extends through the upper 30% to 50% of the image.
            roi_fraction_w (float): Fraction of image width (from corner) to search for pectoral muscle (default: 0.5).
            canny_thresh1 (int): Lower hysteresis threshold for Canny edge detector (default: 30).
            canny_thresh2 (int): Upper hysteresis threshold for Canny edge detector (default: 100).
            fill_value (int): Pixel intensity to replace removed pectoral muscle with (default: 0 = black background).
        """
        # Validate parameter boundaries
        if not (0.1 <= roi_fraction_h <= 1.0):
            raise ValueError(f"roi_fraction_h must be between 0.1 and 1.0, got {roi_fraction_h}")
        if not (0.1 <= roi_fraction_w <= 1.0):
            raise ValueError(f"roi_fraction_w must be between 0.1 and 1.0, got {roi_fraction_w}")
        
        self.roi_fraction_h = roi_fraction_h
        self.roi_fraction_w = roi_fraction_w
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.fill_value = fill_value

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Validate single 2D uint8 grayscale image array dimensions and type.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            np.ndarray: Validated 2D uint8 grayscale array (H, W).
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Input must be a numpy.ndarray, got {type(image)}")
        if image.size == 0:
            raise ValueError("Input image array is empty.")
        if image.dtype != np.uint8:
            raise TypeError(f"Input image must be uint8 (8-bit grayscale), got {image.dtype}")

        # Squeeze trailing channel dimension if input is (H, W, 1)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(axis=2)
        elif image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image (H, W) or (H, W, 1), got shape {image.shape}")

        return image

    def detect_orientation(self, image: np.ndarray) -> str:
        """
        Detect whether the pectoral muscle is positioned in the top-left or top-right corner.

        Mammography Laterality Context:
        - Right Breast MLO view: Pectoral muscle appears in the top-left corner (breast tissue expands to the right).
        - Left Breast MLO view: Pectoral muscle appears in the top-right corner (breast tissue expands to the left).
        This method computes total pixel intensity sum in top-left vs top-right upper quadrants to determine orientation.

        Args:
            image (np.ndarray): 2D uint8 grayscale mammogram.

        Returns:
            str: 'top-left' or 'top-right'.
        """
        img = self._validate_image(image)
        h, w = img.shape
        roi_h = int(h * self.roi_fraction_h)
        roi_w = int(w * self.roi_fraction_w)

        # Calculate integrated intensity in upper-left vs upper-right corner quadrants
        top_left_sum = np.sum(img[:roi_h, :roi_w], dtype=np.float64)
        top_right_sum = np.sum(img[:roi_h, w - roi_w:], dtype=np.float64)

        # Higher intensity sum indicates presence of pectoral muscle + chest wall mass
        return 'top-left' if top_left_sum >= top_right_sum else 'top-right'

    def remove_pectoral_muscle(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and remove pectoral muscle from a single mammogram image.

        Processing Pipeline:
        1. Determine corner orientation ('top-left' vs 'top-right').
        2. Crop upper corner candidate ROI.
        3. Segment high-density region using Otsu automatic thresholding.
        4. Detect candidate boundary edges using Canny edge operator.
        5. Fit straight diagonal line separating muscle from tissue using Probabilistic Hough Transform.
        6. Construct polygon mask covering corner up to muscle border line.
        7. Apply morphological closing to smooth mask edges and erase pectoral muscle.

        Args:
            image (np.ndarray): 2D uint8 grayscale image (H, W).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - cleaned_image (np.ndarray): Image with pectoral muscle zeroed out (same shape H, W).
                - breast_mask (np.ndarray): Binary tissue mask (255 = retained tissue, 0 = removed muscle/background).
        """
        img = self._validate_image(image)
        h, w = img.shape
        
        # Step 1: Detect orientation of pectoral muscle corner
        orientation = self.detect_orientation(img)

        # Compute bounding dimensions for top corner analysis ROI
        roi_h = int(h * self.roi_fraction_h)
        roi_w = int(w * self.roi_fraction_w)

        # Step 2: Extract top corner ROI based on detected orientation
        if orientation == 'top-left':
            roi = img[:roi_h, :roi_w]
        else:
            roi = img[:roi_h, w - roi_w:]

        # Step 3: Otsu automatic thresholding to isolate high-intensity candidate muscle pixels
        # Otsu thresholding finds optimum threshold separating dark background/fat from dense muscle
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 4: Canny Edge Detection to locate sharp boundaries in binarized ROI
        edges = cv2.Canny(thresh, self.canny_thresh1, self.canny_thresh2)

        # Step 5: Probabilistic Hough Line Transform to detect straight diagonal muscle boundary line
        lines = cv2.HoughLinesP(
            edges,
            rho=1,                         # Distance resolution in pixels
            theta=np.pi / 180,              # Angular resolution in radians
            threshold=20,                  # Minimum accumulator votes
            minLineLength=int(min(roi_h, roi_w) * 0.25),  # Minimum line length (25% of ROI dimension)
            maxLineGap=15                  # Maximum gap allowed between line segments
        )

        # Initialize blank binary mask for the ROI region
        muscle_mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)

        line_found = False
        if lines is not None:
            best_line = None
            max_len = 0
            
            # Iterate through all detected line candidates to find the longest diagonal boundary line
            for line in lines:
                x1, y1, x2, y2 = line.ravel()[:4]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                line_len = np.sqrt(dx**2 + dy**2)

                if line_len > max_len:
                    # Compute slope (dy/dx) to verify line is diagonal (not purely vertical or horizontal frame border)
                    slope = dy / max(dx, 1e-5)
                    if 0.2 <= slope <= 5.0:  # Valid diagonal slope range for pectoral muscle boundary
                        max_len = line_len
                        best_line = (x1, y1, x2, y2)

            # Step 6a: If a valid diagonal line was detected, create triangular corner polygon mask
            if best_line is not None:
                x1, y1, x2, y2 = best_line
                line_found = True
                
                # Construct polygon vertices connecting top corner to the diagonal line endpoints
                if orientation == 'top-left':
                    poly_pts = np.array([[0, 0], [x1, y1], [x2, y2], [0, max(y1, y2)]], dtype=np.int32)
                else:
                    poly_pts = np.array([[roi_w, 0], [x1, y1], [x2, y2], [roi_w, max(y1, y2)]], dtype=np.int32)
                
                # Fill polygon region representing pectoral muscle with 255
                cv2.fillPoly(muscle_mask_roi, [poly_pts], 255)

        # Step 6b: Fallback to connected component contour segmentation if line detection yields no candidates
        if not line_found:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Identify contour connected to the top-most corner vertex
                corner_pt = (0, 0) if orientation == 'top-left' else (roi_w - 1, 0)
                for cnt in contours:
                    if cv2.pointPolygonTest(cnt, corner_pt, False) >= 0:
                        cv2.drawContours(muscle_mask_roi, [cnt], -1, 255, -1)
                        break

        # Step 7: Assemble full-sized binary mask matching input image shape (H, W)
        full_muscle_mask = np.zeros((h, w), dtype=np.uint8)
        if orientation == 'top-left':
            full_muscle_mask[:roi_h, :roi_w] = muscle_mask_roi
        else:
            full_muscle_mask[:roi_h, w - roi_w:] = muscle_mask_roi

        # Apply morphological closing filter to eliminate small pinholes and smooth mask boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        full_muscle_mask = cv2.morphologyEx(full_muscle_mask, cv2.MORPH_CLOSE, kernel)

        # Invert mask: 255 represents retained breast tissue, 0 represents removed pectoral muscle + background
        breast_mask = cv2.bitwise_not(full_muscle_mask)

        # Step 8: Erase pectoral muscle pixels in the output mammogram
        cleaned_image = img.copy()
        cleaned_image[full_muscle_mask == 255] = self.fill_value

        return cleaned_image, breast_mask

    def process_batch(self, images: Union[np.ndarray, List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a batch of mammogram images for pectoral muscle removal.

        Args:
            images (Union[np.ndarray, List[np.ndarray]]): 
                Batch array of shape (N, H, W) or (N, H, W, 1), 
                or a list of 2D uint8 numpy arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - cleaned_batch (np.ndarray): Shape (N, H, W) uint8 array of cleaned mammograms.
                - masks_batch (np.ndarray): Shape (N, H, W) uint8 array of binary tissue masks.
        """
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                # Shape (N, H, W)
                results = [self.remove_pectoral_muscle(img) for img in images]
            elif images.ndim == 4 and images.shape[3] == 1:
                # Shape (N, H, W, 1)
                results = [self.remove_pectoral_muscle(img.squeeze(axis=2)) for img in images]
            elif images.ndim == 2:
                # Single 2D image passed to batch processor
                cleaned, mask = self.remove_pectoral_muscle(images)
                return cleaned, mask
            else:
                raise ValueError(f"Unsupported batch array shape: {images.shape}")
        elif isinstance(images, (list, tuple)):
            if len(images) == 0:
                raise ValueError("Input image list is empty.")
            results = [self.remove_pectoral_muscle(img) for img in images]
        else:
            raise TypeError(f"Unsupported batch type: {type(images)}")

        cleaned_list, mask_list = zip(*results)
        return np.stack(cleaned_list, axis=0), np.stack(mask_list, axis=0)

    def visualize_pectoral_removal(
        self,
        image: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5)
    ) -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
        """
        Generate visual comparison plot: Original Mammogram vs. Retained Tissue Mask vs. Pectoral Removed.

        Args:
            image (np.ndarray): Original input mammogram (H, W).
            save_path (Optional[str]): File path to save output visualization image.
            figsize (Tuple[int, int]): Size of matplotlib figure layout.

        Returns:
            Tuple[np.ndarray, np.ndarray, plt.Figure]:
                (cleaned_image, breast_mask, figure_handle)
        """
        img = self._validate_image(image)
        cleaned_img, mask = self.remove_pectoral_muscle(img)
        orientation = self.detect_orientation(img)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Panel 1: Original input image
        axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title(f"1. Original Mammogram ({orientation.capitalize()})", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Panel 2: Retained breast tissue mask
        axes[1].imshow(mask, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title("2. Retained Tissue Mask", fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Panel 3: Cleaned mammogram image with pectoral muscle erased
        axes[2].imshow(cleaned_img, cmap='gray', vmin=0, vmax=255)
        axes[2].set_title("3. Pectoral Muscle Removed", fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        return cleaned_img, mask, fig


# Functional wrapper interface matching pipeline specifications
def remove_pectoral_muscle_roi(
    image: np.ndarray,
    roi_fraction_h: float = 0.5,
    roi_fraction_w: float = 0.5,
    fill_value: int = 0
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function interface to remove pectoral muscle from single images or image batches.

    Args:
        image (np.ndarray): 8-bit uint8 grayscale image (H, W) or batch (N, H, W).
        roi_fraction_h (float): Fractional height ROI search boundary (default: 0.5).
        roi_fraction_w (float): Fractional width ROI search boundary (default: 0.5).
        fill_value (int): Replacement pixel value for removed pectoral muscle (default: 0).

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Cleaned image or batch array.
    """
    remover = PectoralMuscleRemover(
        roi_fraction_h=roi_fraction_h,
        roi_fraction_w=roi_fraction_w,
        fill_value=fill_value
    )
    if image.ndim in (3, 4) and (image.shape[0] > 1 or image.ndim == 4):
        cleaned_batch, _ = remover.process_batch(image)
        return cleaned_batch
    cleaned, _ = remover.remove_pectoral_muscle(image)
    return cleaned
