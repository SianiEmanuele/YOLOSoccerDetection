import cv2
import gc
import torch
import os
import yaml
import glob
from tqdm import tqdm

# Environment settings and imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from ultralytics.models.yolo import YOLO
from torch.utils.data import Dataset, DataLoader
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer


# ==============================================================================
# 1. DATASET CLASS WITH CACHING LOGIC
# ==============================================================================
class CachedSRYOLODataset(Dataset):
    """
    A Dataset that creates a cache of pre-processed images with Super-Resolution.
    Performs SR only once and then loads data from cache for fast training.
    """

    def __init__(self, img_dir, label_dir, sryolo_instance, cache_dir_name="cache", force_recache=False):
        super().__init__()

        # The SRYOLO instance is needed to use its SR method
        self.sryolo_instance = sryolo_instance

        # Build paths for cache
        # Cache is created relative to the dataset folder
        base_dir = os.path.dirname(img_dir)
        self.cache_dir = os.path.join(base_dir, cache_dir_name)
        self.cached_img_dir = os.path.join(self.cache_dir, "images")
        self.cached_label_dir = os.path.join(self.cache_dir, "labels")

        self.image_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))  # or other formats
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, '*.txt')))

        # Check if cache exists and needs to be created
        num_cached_imgs = len(glob.glob(os.path.join(self.cached_img_dir, '*.pt')))
        if force_recache or num_cached_imgs != len(self.image_paths):
            print(f"Cache not found or incomplete. Creating cache in: {self.cache_dir}")
            self.create_cache(img_dir, label_dir)
        else:
            print(f"Valid cache found. Loading from: {self.cache_dir}")

        # Now, point to files in cache
        self.cached_image_files = sorted(glob.glob(os.path.join(self.cached_img_dir, '*.pt')))
        self.cached_label_files = sorted(glob.glob(os.path.join(self.cached_label_dir, '*.txt')))

    def create_cache(self, img_dir, label_dir):
        os.makedirs(self.cached_img_dir, exist_ok=True)
        os.makedirs(self.cached_label_dir, exist_ok=True)

        print("Applying Super-Resolution to create cache. This may take time...")
        for img_path in tqdm(self.image_paths, desc="Caching Images"):
            base_filename = os.path.splitext(os.path.basename(img_path))[0]

            # Load image as NumPy array, as your method does
            img_np = cv2.imread(img_path)
            img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            sr_np_rgb = self.sryolo_instance.apply_sr_np(img_np_rgb)

            # Save visible image for verification
            cv2.imwrite(os.path.join(self.cached_img_dir, f"{base_filename}.png"),
                        cv2.cvtColor(sr_np_rgb, cv2.COLOR_RGB2BGR))

            # Convert to tensor and save for training
            sr_tensor = torch.from_numpy(sr_np_rgb.transpose(2, 0, 1)).float() / 255.0
            torch.save(sr_tensor, os.path.join(self.cached_img_dir, f"{base_filename}.pt"))

            # Copy corresponding label file
            label_path = os.path.join(label_dir, f"{base_filename}.txt")
            if os.path.exists(label_path):
                import shutil
                shutil.copy(label_path, self.cached_label_dir)

    def __len__(self):
        return len(self.cached_image_files)

    def __getitem__(self, index):
        # Load tensor and labels from cache
        img_tensor = torch.load(self.cached_image_files[index])
        pass  # Actual logic is handled in SRYOLO train method


# ==============================================================================
# 2. MODIFIED SRYOLO CLASS
# ==============================================================================
class SRYOLO(YOLO):
    def __init__(self, yolo_weights, upscale=4, gan_weights=None, dni_weight=0.5,
                 tile=0, tile_pad=10, pre_pad=0):

        super().__init__(model=yolo_weights)
        self.yolo_weights_path = yolo_weights
        self._sr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device for Super-Resolution: {self._sr_device}")

        if gan_weights:
            arch = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=upscale,
                                   act_type='prelu')
            self.sr_model = RealESRGANer(
                scale=upscale, model_path=gan_weights, dni_weight=dni_weight, model=arch,
                tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, half=True,
                gpu_id=0 if 'cuda' in str(self._sr_device) else -1
            )
            print("Super-Resolution model loaded successfully. 👍")
        else:
            self.sr_model = None
            print("No GAN weights provided, Super-Resolution is disabled.")

    def apply_sr_np(self, img_np_rgb):
        if not self.sr_model:
            return img_np_rgb
        original_height, original_width = img_np_rgb.shape[:2]
        original_dims = (original_width, original_height)
        try:
            with torch.no_grad():
                enhanced_output, _ = self.sr_model.enhance(img_np_rgb, outscale=self.sr_model.scale)
            resized_output = cv2.resize(enhanced_output, original_dims, interpolation=cv2.INTER_AREA)
            del enhanced_output
            gc.collect()
            torch.cuda.empty_cache()
            return resized_output
        except Exception as e:
            print(f"[SR ERROR] Error during Super-Resolution application: {e}")
            return img_np_rgb

    def _create_dataset_cache(self, img_path, label_path, cache_name):
        """Helper to create cache for a given split (train/val)."""
        print(f"\n--- Checking cache for split in {img_path} ---")
        cache_img_path = os.path.join(os.path.dirname(img_path), cache_name, "images")

        num_orig_imgs = len(glob.glob(os.path.join(img_path, '*')))
        num_cached_imgs = len(glob.glob(os.path.join(cache_img_path, '*.png')))

        if num_cached_imgs == num_orig_imgs:
            print("Cache found and complete. Training will use images from cache.")
            return

        print("Cache not found or incomplete. Starting creation...")
        os.makedirs(cache_img_path, exist_ok=True)
        cache_label_path = os.path.join(os.path.dirname(label_path), cache_name, "labels")
        os.makedirs(cache_label_path, exist_ok=True)

        original_image_files = sorted(glob.glob(os.path.join(img_path, '*')))
        for img_file in tqdm(original_image_files, desc=f"Caching {cache_name}"):
            base_filename = os.path.splitext(os.path.basename(img_file))[0]

            img_np = cv2.imread(img_file)
            img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            sr_np_rgb = self.apply_sr_np(img_np_rgb)

            # Save processed image
            cv2.imwrite(os.path.join(cache_img_path, f"{base_filename}.png"),
                        cv2.cvtColor(sr_np_rgb, cv2.COLOR_RGB2BGR))

            # Copy label file
            orig_label_file = os.path.join(label_path, f"{base_filename}.txt")
            if os.path.exists(orig_label_file):
                import shutil
                shutil.copy(orig_label_file, cache_label_path)

    def train(self, **kwargs):
        """
        Performs training after creating (if necessary) a pre-processed dataset
        in a clean cache folder, with robust file search.
        """
        # 1. Read data.yaml file to get paths
        data_yaml_path = kwargs['data']
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        # Base path of dataset, relative to .yaml file
        base_path = os.path.abspath(os.path.dirname(data_yaml_path))

        cached_data_config = data_config.copy()
        splits_to_cache = {'train', 'val'}

        for split in splits_to_cache:
            if split not in data_config:
                continue

            # fixed discrepancy in 'val' split name
            if split == 'val':
                split = 'valid'

            original_img_dir = os.path.join(base_path, split, 'images')
            cache_dir = os.path.join(base_path, f"cache_{split}")
            cached_img_dir = os.path.join(cache_dir, "images")
            cached_label_dir = os.path.join(cache_dir, "labels")

            print(f"\n--- Checking cache for split '{split}' ---")

            # DEBUG print to verify path
            print(f"Looking for original images in: {original_img_dir}")

            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
            original_image_files = []
            for ext in image_extensions:
                original_image_files.extend(glob.glob(os.path.join(original_img_dir, ext)))

            # Remove any duplicates and sort
            original_image_files = sorted(list(set(original_image_files)))
            num_orig_imgs = len(original_image_files)

            # Another DEBUG print
            print(f"Found {num_orig_imgs} original image files.")

            if num_orig_imgs == 0:
                print(
                    "WARNING: No image files found. Check that the path printed above is correct and contains images with valid extensions.")
                continue  # Move to next split (e.g. 'val')

            num_cached_imgs = len(glob.glob(os.path.join(cached_img_dir, '*.png')))

            if num_cached_imgs != num_orig_imgs:
                print(f"Cache for '{split}' not found or incomplete. Creating...")
                os.makedirs(cached_img_dir, exist_ok=True)
                os.makedirs(cached_label_dir, exist_ok=True)

                original_label_dir = original_img_dir.replace('images', 'labels')

                for img_file in tqdm(original_image_files, desc=f"Caching {split}"):
                    base_filename = os.path.splitext(os.path.basename(img_file))[0]

                    img_np = cv2.imread(img_file)
                    img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

                    sr_np_rgb = self.apply_sr_np(img_np_rgb)

                    cv2.imwrite(os.path.join(cached_img_dir, f"{base_filename}.png"),
                                cv2.cvtColor(sr_np_rgb, cv2.COLOR_RGB2BGR))

                    orig_label_file = os.path.join(original_label_dir, f"{base_filename}.txt")
                    if os.path.exists(orig_label_file):
                        import shutil
                        shutil.copy(orig_label_file, cached_label_dir)
            else:
                print(f"Cache for '{split}' found and complete.")

            # fixed discrepancy in 'val' split name
            if split == 'valid':
                split = 'val'

            cached_data_config[split] = os.path.relpath(cached_img_dir, base_path).replace('\\', '/')

        # 3. Create new data.yaml file pointing to cache
        cached_yaml_path = os.path.join(base_path, 'cached_data.yaml')
        with open(cached_yaml_path, 'w') as f:
            yaml.dump(cached_data_config, f)

        print(f"\nTraining started using cached dataset specified in: {cached_yaml_path}")

        if self.sr_model:
            del self.sr_model
            self.sr_model = None
            gc.collect()
            torch.cuda.empty_cache()
            print("\nSR model released from memory. Starting YOLO training.")

        kwargs['data'] = cached_yaml_path
        super().train(**kwargs)

    def predict(self, source=None, **kwargs):
        return super().predict(source=source, **kwargs)


# ==============================================================================
# 3. USAGE EXAMPLE
# ==============================================================================
if __name__ == '__main__':
    cwd = os.getcwd()
    gan_weights = os.path.join(cwd, "..", "esrgan", "pretrained", "realesr-general-x4v3.pth")

    # Create SRYOLO model with on-the-fly SR
    sr_yolo = SRYOLO(
        yolo_weights="yolov9c.pt",
        upscale=4,
        gan_weights=gan_weights,
        dni_weight=0.5,
        tile=0,
        tile_pad=10,
        pre_pad=0,
    )
    sr_yolo = sr_yolo.cuda()

    # Training with on-the-fly SR
    dataset_path = os.path.join(cwd, "..", "..", "..", "dataset", "yolov9", "v0")

    # Method 1: Direct training (recommended)
    sr_yolo.train(
        data=os.path.join(dataset_path, "data.yaml"),
        epochs=50,
        imgsz=1280,
        save=True,
        project="yolo_football_analysis",
        name="yoloSR_dataset_v3_high_res_onthefly",
        batch=4
    )