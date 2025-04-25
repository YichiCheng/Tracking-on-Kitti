import torch
import torchvision.transforms as T
import cv2
from torch import nn
from PIL import Image

class ReIDExtractor:
    def __init__(self, model_weights_path=None, device='cuda'):
        # MobileNetV2 作为特征提取器
        from torchvision.models import mobilenet_v2

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = mobilenet_v2(pretrained=True)


        self.model = nn.Sequential(*list(self.model.features.children()))
        self.model.eval()
        self.model.to(self.device)

        if model_weights_path is not None:
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))

        self.transform = T.Compose([
            T.Resize((128, 256)),  # 常见ReID输入尺寸
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准化
        ])

    def extract(self, img_crop):
        """
        img_crop: 输入单张小图 (numpy array)，一般是BGR格式（cv2读取）
        返回：提取后的特征向量 (numpy array)
        """
        if img_crop is None or img_crop.size == 0:
            return None

        # OpenCV读取是BGR，需要转RGB
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        # 再转成PIL Image
        img_crop = Image.fromarray(img_crop)

        img = self.transform(img_crop).unsqueeze(0).to(self.device)  # 加batch维度
        with torch.no_grad():
            feature = self.model(img)
        feature = feature.view(feature.size(0), -1)  # 拉成一维
        feature = nn.functional.normalize(feature, dim=1)  # 单位归一化
        return feature.squeeze(0).cpu().numpy()
