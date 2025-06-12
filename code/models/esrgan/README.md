# Finetuning `realesr-general-x4v3`

This guide provides step-by-step instructions to finetune the `realesr-general-x4v3` model using your own high-resolution images.

---

## 🔧 How to Finetune

1. **Prepare Dataset**  
   Place your custom high-resolution images in the following folder:  
   ```
   src/models/esrgan/train/dataset/input_dataset
   ```
   > ⚠️ You only need to provide **high-resolution** images. Low-resolution counterparts will be generated automatically.

2. **Generate Multi-scale Images**  
   Run the following script to create multiple resolution versions of the dataset:
   ```bash
   python generate_multiscale_DF2K.py
   ```

3. **Extract Sub-images**  
   To break the images into smaller patches for training:
   ```bash
   python extract_subimages.py
   ```

4. **Generate Meta Information**  
   This script prepares metadata needed for the training pipeline:
   ```bash
   python generate_meta_info.py
   ```

5. **Modify finetuning parameters**
    - You can customize the finetuning process by modifying the following file:
    ```bash
    src\models\esrgan\options\finetune_realesr-general-x4v3.yml    
    ```

5. **Start Training**  
   Launch the training process:
   ```bash
   python src/models/esrgan/realesrgan/train.py
   ```

---

## 🧪 How to Test

1. **Update Input Path**  
   In `GAN_test.py`, modify the `--input` argument to point to your test image or folder:
   ```python
    parser.add_argument('-i', '--input', type=str, default='src\models\esrgan\input.jpg', help='Input image or folder')
   ```

2. **Run the Test Script**  
   Execute:
   ```bash
   python GAN_test.py
   ```

---
