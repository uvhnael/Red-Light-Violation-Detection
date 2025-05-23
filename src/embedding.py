import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
import numpy as np
import cv2

class EmbeddingExtractor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = mobilenet_v2(pretrained=True).features.eval().to(self.device)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding vector from an image patch (BGR image).
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor).mean([2, 3])  # Global Average Pooling
        return embedding.cpu().numpy().flatten().astype(np.float32)
