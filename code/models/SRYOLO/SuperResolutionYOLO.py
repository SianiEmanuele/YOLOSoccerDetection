import os
import cv2
import torch
import numpy as np
from torch import nn
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer
from ultralytics import YOLO
import gc
from pathlib import Path

class SRWrapper(nn.Module):
    """
    Applicazione on-the-fly di Real-ESRGAN come primo layer.
    """
    def __init__(self, upsampler: RealESRGANer, max_size: int, stride: int):
        super().__init__()
        self.upsampler = upsampler
        self.max_size = max_size
        self.stride = stride

        # ultralytics compatibility for prediction
        self.f = -1 # avoid warning about unused variable
        self.i = 0

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: BxCxHxW, valori in [0,1]
        b, c, h0, w0 = x.shape
        out = []
        for i in range(b):
            img = (x[i].cpu().permute(1,2,0).numpy() * 255).astype('uint8')
            try:
                sr, _ = self.upsampler.enhance(img, outscale=1)
            except RuntimeError as e:
                print(f"OOM during SR: {e}")
                torch.cuda.empty_cache()
                sr = img
            torch.cuda.empty_cache()
            # resize
            h, w = sr.shape[:2]
            scale_ratio = self.max_size / max(h, w)
            new_h, new_w = int(h*scale_ratio), int(w*scale_ratio)
            sr = cv2.resize(sr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # pad
            ph, pw = (-new_h) % self.stride, (-new_w) % self.stride
            top, bottom = ph//2, ph-ph//2
            left, right = pw//2, pw-pw//2
            sr = np.pad(sr, ((top,bottom),(left,right),(0,0)), constant_values=114)
            tensor_sr = torch.from_numpy(sr).permute(2,0,1).float()/255.0
            out.append(tensor_sr.to(x.device))
        return torch.stack(out)

class SRYOLO(nn.Module):
    """
    Integrazione di Real-ESRGAN nel modello YOLOv9c.
    SR viene applicata on-the-fly durante train, val e predict.
    """
    def __init__(
        self,
        yolo_weights: str,
        scale: int,
        model_path: str,
        dni_weight: float,
        tile: int,
        tile_pad: int,
        pre_pad: int,
        max_size: int = 640,
        device: str = 'cuda:0'
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_size = max_size
        # Init SR upsampler
        arch = SRVGGNetCompact(3,3,64,32,scale,'prelu')
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=arch,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=True,
            gpu_id=0 if 'cuda' in device else -1
        )
        # Init YOLO
        self.yolo = YOLO(yolo_weights)

        if not self.is_trained_model(yolo_weights):
            # Se stai usando un modello base tipo yolov9c.pt, allora aggiungi SRWrapper
            stride = int(self.yolo.model.stride.max())
            self.yolo.model.model = nn.Sequential(
                SRWrapper(self.upsampler, self.max_size, stride),
                *list(self.yolo.model.model.children())
            )
        # Altrimenti, `best.pt` si assume già abbia SRWrapper incluso nel backbone

        self.yolo.model.to(self.device)

    def is_trained_model(self, path: str) -> bool:
        return Path(path).name in ['best.pt', 'last.pt']

    def train(self, **kwargs):
        """Chiamata identica a YOLO.train, con SR on-the-fly integrata."""
        return self.yolo.train(**kwargs)

    def val(self, **kwargs):
        return self.yolo.val(**kwargs)

    def predict(self, source, **kwargs):
        """
        Predict con SR integrata e gestione memoria ottimizzata
        """
        # Imposta dimensione immagine se non specificata
        if 'imgsz' not in kwargs:
            kwargs['imgsz'] = self.max_size

        # Riduci batch size per evitare OOM durante predict
        original_batch = kwargs.get('batch', None)
        if original_batch is None or original_batch > 4:
            kwargs['batch'] = 2

        try:
            # Pulizia memoria prima della predict
            torch.cuda.empty_cache()
            gc.collect()

            # Esegui predict
            results = self.yolo.predict(source, **kwargs)

            # Filtro dei risultati: rimuovi quelli con immagini non valide
            valid_results = []
            for r in results:
                if r.orig_img is not None and isinstance(r.orig_img, np.ndarray):
                    valid_results.append(r)
                else:
                    print(f"Warning: Result with None image skipped.")

            return valid_results

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"OOM durante predict, riprovo con batch=1: {e}")
            torch.cuda.empty_cache()
            gc.collect()

            kwargs['batch'] = 1
            results = self.yolo.predict(source, **kwargs)

            valid_results = []
            for r in results:
                if r.orig_img is not None and isinstance(r.orig_img, np.ndarray):
                    valid_results.append(r)
                else:
                    print(f"Warning: Result with None image skipped.")

            return valid_results

        finally:
            torch.cuda.empty_cache()
            gc.collect()



# Esempio
if __name__ == '__main__':
    import os
    dataset_path = r'dataset\yolov9\v3'
    sr_yolo = SRYOLO(
        yolo_weights='yolov9c.pt',
        scale=4,
        model_path=r'src\models\esrgan\experiments\finetune_Realesr-general-x4v3_2\models\net_g_latest.pth',
        dni_weight=0.5,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        max_size=1280
    )
    sr_yolo.train(
        data=os.path.join(dataset_path, 'data.yaml'),
        epochs=50,
        imgsz=1280,
        save=True,
        project="yolo_football_analysis",
        name="yoloSR_dataset_v3_high_res",
        batch=4
    )
    sr_yolo.val(data=os.path.join(dataset_path, 'data.yaml'), imgsz=1280)
    preds = sr_yolo.predict(source='dataset/images', imgsz=1280)
    for r in preds:
        print(r.orig_img.shape, len(r.boxes))
