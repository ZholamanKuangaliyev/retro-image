import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm

class HOGDescriptor:
    """
    Histogram of Oriented Gradients (HOG) descriptor implementation
    Based on: Dalal & Triggs, "Histograms of Oriented Gradients for Human Detection", CVPR 2005
    
    Default parameters follow the paper's optimal settings for human detection:
    - Cell size: 8x8 pixels
    - Block size: 2x2 cells (16x16 pixels)
    - Block stride: 8 pixels (50% overlap)
    - Orientation bins: 9 (0-180 degrees, unsigned)
    - Detection window: 64x128 pixels
    """
    
    def __init__(self, 
                 cell_size=8,
                 block_size=2,  # in cells
                 block_stride=8,  # in pixels
                 n_bins=9,
                 signed_gradient=False,
                 gamma_normalize=False,
                 L2_Hys_threshold=0.2):
        
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.n_bins = n_bins
        self.signed_gradient = signed_gradient
        self.gamma_normalize = gamma_normalize
        self.L2_Hys_threshold = L2_Hys_threshold
        
        # Orientation range: 0-180 (unsigned) or 0-360 (signed)
        self.max_angle = 360 if signed_gradient else 180
        
    def _gamma_normalize(self, image):
        """Apply gamma compression (square root) to reduce shadow effects"""
        if not self.gamma_normalize:
            return image
        # Square root compression
        return np.sqrt(image / 255.0) * 255.0
    
    def _compute_gradients(self, image):
        """
        Compute gradients using simple 1D [-1, 0, 1] masks at sigma=0
        This is the optimal setting according to the paper (Section 6.2)
        """
        # Convert to float and normalize
        if len(image.shape) == 3:
            # For color images, compute gradient for each channel and take max
            gx = np.zeros_like(image, dtype=np.float32)
            gy = np.zeros_like(image, dtype=np.float32)
            
            for c in range(image.shape[2]):
                gx[:,:,c] = convolve1d(image[:,:,c].astype(np.float32), [-1, 0, 1], axis=1, mode='nearest')
                gy[:,:,c] = convolve1d(image[:,:,c].astype(np.float32), [-1, 0, 1], axis=0, mode='nearest')
            
            # Take gradient with largest magnitude across channels
            magnitude_c = np.sqrt(gx**2 + gy**2)
            max_c = np.argmax(magnitude_c, axis=2)
            
            # Select gradients from dominant channel
            rows, cols = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
            gx = gx[rows, cols, max_c]
            gy = gy[rows, cols, max_c]
        else:
            # Grayscale
            image = image.astype(np.float32)
            gx = convolve1d(image, [-1, 0, 1], axis=1, mode='nearest')
            gy = convolve1d(image, [-1, 0, 1], axis=0, mode='nearest')
        
        # Gradient magnitude and orientation
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * 180 / np.pi  # -180 to 180
        
        # Convert to unsigned (0-180) if needed
        if not self.signed_gradient:
            orientation = np.abs(orientation)
        
        return magnitude, orientation
    
    def _cell_histogram(self, magnitude, orientation, cell_x, cell_y):
        """
        Compute histogram for a single cell using bilinear interpolation
        """
        # Extract cell region
        x_start = cell_x * self.cell_size
        y_start = cell_y * self.cell_size
        x_end = min(x_start + self.cell_size, magnitude.shape[1])
        y_end = min(y_start + self.cell_size, magnitude.shape[0])
        
        mag_cell = magnitude[y_start:y_end, x_start:x_end]
        ori_cell = orientation[y_start:y_end, x_start:x_end]
        
        histogram = np.zeros(self.n_bins)
        bin_width = self.max_angle / self.n_bins
        
        # Bilinear interpolation voting
        for y in range(mag_cell.shape[0]):
            for x in range(mag_cell.shape[1]):
                mag = mag_cell[y, x]
                angle = ori_cell[y, x]
                
                # Handle edge case at max_angle
                if angle >= self.max_angle:
                    angle = self.max_angle - 0.001
                
                # Compute bin positions with interpolation
                bin_pos = angle / bin_width
                bin_low = int(np.floor(bin_pos)) % self.n_bins
                bin_high = (bin_low + 1) % self.n_bins
                weight_high = bin_pos - bin_low
                weight_low = 1 - weight_high
                
                # Vote with gradient magnitude
                histogram[bin_low] += weight_low * mag
                histogram[bin_high] += weight_high * mag
        
        return histogram
    
    def _compute_cell_histograms(self, magnitude, orientation):
        """Compute histograms for all cells"""
        n_cells_x = magnitude.shape[1] // self.cell_size
        n_cells_y = magnitude.shape[0] // self.cell_size
        
        cell_histograms = np.zeros((n_cells_y, n_cells_x, self.n_bins))
        
        for y in range(n_cells_y):
            for x in range(n_cells_x):
                cell_histograms[y, x] = self._cell_histogram(magnitude, orientation, x, y)
        
        return cell_histograms
    
    def _normalize_block(self, block_features):
        """
        L2-Hys normalization: L2 norm, clip, then renormalize
        This is the best performing normalization according to the paper (Section 6.4)
        """
        # L2 norm
        eps = 1e-7
        norm = np.sqrt(np.sum(block_features**2) + eps**2)
        normalized = block_features / norm
        
        # Clip (Hysteresis)
        normalized = np.clip(normalized, 0, self.L2_Hys_threshold)
        
        # Renormalize
        norm = np.sqrt(np.sum(normalized**2) + eps**2)
        normalized = normalized / norm
        
        return normalized
    
    def _compute_block_features(self, cell_histograms):
        """
        Concatenate and normalize overlapping blocks
        """
        n_cells_y, n_cells_x = cell_histograms.shape[:2]
        
        # Calculate number of blocks
        n_blocks_x = (n_cells_x - self.block_size) * self.cell_size // self.block_stride + 1
        n_blocks_y = (n_cells_y - self.block_size) * self.cell_size // self.block_stride + 1
        
        features = []
        block_positions = []
        
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                # Convert pixel stride to cell indices
                cell_stride = self.block_stride // self.cell_size
                x_start = bx * cell_stride
                y_start = by * cell_stride
                
                # Extract block (2x2 cells by default)
                block = cell_histograms[y_start:y_start+self.block_size, 
                                       x_start:x_start+self.block_size]
                
                # Flatten block
                block_vector = block.flatten()
                
                # Normalize
                normalized = self._normalize_block(block_vector)
                
                features.append(normalized)
                block_positions.append((x_start, y_start))
        
        return np.concatenate(features), block_positions
    
    def compute(self, image):
        """
        Compute HOG features for an image
        
        Returns:
            features: 1D array of HOG features
            visualization_data: dict with intermediate results for visualization
        """
        # Step 1: Optional gamma normalization
        image_normalized = self._gamma_normalize(image)
        
        # Step 2: Compute gradients
        magnitude, orientation = self._compute_gradients(image_normalized)
        
        # Step 3: Compute cell histograms
        cell_histograms = self._compute_cell_histograms(magnitude, orientation)
        
        # Step 4: Normalize and concatenate blocks
        features, block_positions = self._compute_block_features(cell_histograms)
        
        visualization_data = {
            'image': image,
            'magnitude': magnitude,
            'orientation': orientation,
            'cell_histograms': cell_histograms,
            'block_positions': block_positions
        }
        
        return features, visualization_data
    
    def visualize(self, visualization_data, figsize=(16, 4)):
        """
        Visualize HOG computation steps
        """
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Original image
        ax = axes[0]
        if len(visualization_data['image'].shape) == 3:
            ax.imshow(visualization_data['image'].astype(np.uint8))
        else:
            ax.imshow(visualization_data['image'], cmap='gray')
        ax.set_title('Input Image')
        ax.axis('off')
        
        # Gradient magnitude
        ax = axes[1]
        ax.imshow(visualization_data['magnitude'], cmap='hot')
        ax.set_title('Gradient Magnitude')
        ax.axis('off')
        
        # Gradient orientation
        ax = axes[2]
        ax.imshow(visualization_data['orientation'], cmap='hsv')
        ax.set_title('Gradient Orientation')
        ax.axis('off')
        
        # HOG visualization (dominant orientations)
        ax = axes[3]
        self._plot_hog_cells(ax, visualization_data['cell_histograms'])
        ax.set_title('HOG Cells (Dominant Orientations)')
        
        plt.tight_layout()
        return fig
    
    def _plot_hog_cells(self, ax, cell_histograms):
        """Visualize HOG cells as line segments showing dominant orientations"""
        n_cells_y, n_cells_x = cell_histograms.shape[:2]
        
        # Create overlay image
        cell_img = np.zeros((n_cells_y * self.cell_size, n_cells_x * self.cell_size))
        ax.imshow(cell_img, cmap='gray', vmin=0, vmax=1)
        
        bin_width = self.max_angle / self.n_bins
        
        for y in range(n_cells_y):
            for x in range(n_cells_x):
                center_y = y * self.cell_size + self.cell_size // 2
                center_x = x * self.cell_size + self.cell_size // 2
                
                histogram = cell_histograms[y, x]
                
                # Draw lines for each orientation bin weighted by magnitude
                for bin_idx, strength in enumerate(histogram):
                    if strength > 0:
                        angle = bin_idx * bin_width + bin_width / 2
                        angle_rad = np.radians(angle)
                        
                        # Line length proportional to histogram value
                        length = (strength / (np.max(histogram) + 1e-7)) * self.cell_size * 0.4
                        
                        dx = length * np.cos(angle_rad)
                        dy = length * np.sin(angle_rad)
                        
                        ax.plot([center_x - dx, center_x + dx], 
                               [center_y - dy, center_y + dy],
                               'w-', linewidth=1.5, alpha=min(strength / 10, 1.0))
        
        ax.set_xlim(0, n_cells_x * self.cell_size)
        ax.set_ylim(n_cells_y * self.cell_size, 0)
        ax.axis('off')


def load_image(path):
    """Load an image from disk using matplotlib (returns uint8 numpy array)"""
    img = plt.imread(path)
    # If float (png), convert to uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return img


def resize_image(image, target_height):
    """Resize image to target height preserving aspect ratio using nearest neighbor"""
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    new_h = target_height

    row_indices = (np.arange(new_h) / scale).astype(int)
    col_indices = (np.arange(new_w) / scale).astype(int)
    row_indices = np.clip(row_indices, 0, h - 1)
    col_indices = np.clip(col_indices, 0, w - 1)

    if len(image.shape) == 3:
        return image[np.ix_(row_indices, col_indices, np.arange(image.shape[2]))]
    return image[np.ix_(row_indices, col_indices)]


def rgb_to_gray(image):
    """Convert RGB to grayscale using luminance weights"""
    if len(image.shape) == 2:
        return image
    return (0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]).astype(np.uint8)


def demo():
    """Demonstrate HOG descriptor on a real human photo"""
    print("=" * 60)
    print("HOG (Histogram of Oriented Gradients) Implementation")
    print("Based on Dalal & Triggs, CVPR 2005")
    print("=" * 60)

    # Load real photo
    img_path = 'data/man1.jpeg'
    print(f"\nLoading image: {img_path}")
    image = load_image(img_path)
    print(f"Original image shape: {image.shape}")

    # Resize to a reasonable detection window height (keep aspect ratio)
    # The paper uses 64x128 windows, but we process the whole image
    target_h = 512
    image = resize_image(image, target_h)
    print(f"Resized image shape: {image.shape}")

    # Initialize HOG with paper's optimal parameters
    print("\nHOG parameters (Dalal & Triggs optimal):")
    print("  - Cell size: 8x8 pixels")
    print("  - Block size: 2x2 cells (16x16 pixels)")
    print("  - Block stride: 8 pixels (50% overlap)")
    print("  - Orientation bins: 9 (0-180 degrees, unsigned)")
    print("  - Normalization: L2-Hys (clip at 0.2)")

    hog = HOGDescriptor(
        cell_size=8,
        block_size=2,
        block_stride=8,
        n_bins=9,
        signed_gradient=False,
        gamma_normalize=True,
        L2_Hys_threshold=0.2,
    )

    # Compute features
    print("\nComputing HOG features...")
    features, vis_data = hog.compute(image)

    n_cells_x = image.shape[1] // 8
    n_cells_y = image.shape[0] // 8
    n_blocks_x = (n_cells_x - 2) * 8 // 8 + 1
    n_blocks_y = (n_cells_y - 2) * 8 // 8 + 1

    print(f"  Cells: {n_cells_x} x {n_cells_y}")
    print(f"  Blocks: {n_blocks_x} x {n_blocks_y}")
    print(f"  Feature vector length: {len(features)}")

    # Feature statistics
    print(f"\nFeature statistics:")
    print(f"  Mean: {np.mean(features):.4f}")
    print(f"  Std:  {np.std(features):.4f}")
    print(f"  Min:  {np.min(features):.4f}")
    print(f"  Max:  {np.max(features):.4f}")

    # Visualize
    print("\nGenerating visualization...")
    fig = hog.visualize(vis_data, figsize=(20, 5))
    plt.savefig('output/hog_demo.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'output/hog_demo.png'")
    plt.show()

    return hog, features, vis_data


if __name__ == "__main__":
    hog, features, vis_data = demo()