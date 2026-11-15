
#asking IA from Bing "python train labels images with fpn"
# slightly modified by Alfonso Blanco

import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import json

# -----------------------------
# 1. Custom Dataset Class
# -----------------------------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms

        # Load COCO-style annotations
        with open(annotations_file) as f:
            coco_data = json.load(f)

        self.images_info = coco_data["images"]
        self.annotations = coco_data["annotations"]

        # Map image_id -> list of annotations
        self.image_to_anns = {}
        for ann in self.annotations:
            self.image_to_anns.setdefault(ann["image_id"], []).append(ann)

        # Category mapping
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    def __getitem__(self, idx):
        img_info = self.images_info[idx]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        anns = self.image_to_anns.get(img_info["id"], [])

        boxes = []
        labels = []
        areas=[]
        for ann in anns:
            xmin, ymin, w, h = ann["bbox"]
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            
            #labels.append(ann["category_id"])
            # --- CORRECCIÓN AQUÍ --- De la IA de Google
            # Sumamos 1 para que las clases (0 y 1) pasen a ser (1 y 2).
            # De esta forma el 0 queda reservado de manera segura para el fondo (background).
            labels.append(ann["category_id"] + 1)
            areas.append(ann["area"])


         # Si una imagen no tiene anotaciones válidas, creamos tensores vacíos con las dimensiones correctas
         # sugerido por la IA
         #  ya que si  JSON tiene algún error menor en una caja, Faster R-CNN arrojará un error de entrenamiento.
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([img_info["id"]])
            area = torch.as_tensor([ann["area"] for ann in anns], dtype=torch.float32)
            iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        #return img, target
        # Cambio indicado por la IA de google (09/11/2026)
        #if self.transforms:
            # v2 transforma la imagen y las cajas al mismo tiempo
        #   img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images_info)

# -----------------------------
# 2. Data Transforms
# -----------------------------
def get_transform(train):
    transforms = []
    transforms.append(F.to_tensor)
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

"""
Comentarios IA Bing
Tus transforms actuales funcionan perfectamente para inferencia (test/evaluación),
pero si planeas entrenar el modelo, usar torchvision.transforms.ToTensor() directamente desde torchvision.transforms.Compose
romperá el entrenamiento si en el futuro decides añadir aumentación de datos (como volteos horizontales, rotaciones o
cambios de brillo).
Esto ocurre porque en la detección de objetos de torchvision, las transformaciones deben aplicarse de manera simultánea
tanto a la imagen como a las cajas de texto (target["boxes"]) para que se mantengan alineadas.
Para corregirlo y dejar tu código preparado tanto para entrenamiento como para pruebas de forma robusta,
debes importar el módulo de transforms específico de detección de torchvision.transforms.v2.
"""
from torchvision.transforms import v2
def get_transformRecomendadaPorIA(train=True):
    transforms = []
    
    # Convierte la imagen a tensor y escala sus píxeles de 0 a 1 de forma segura para detección
    transforms.append(v2.ToImage())
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    
    if train:
        # Aquí puedes añadir aumentación de datos de forma segura para el futuro
        # Ejemplo: Volteo horizontal aleatorio (afecta tanto a imagen como a cajas automáticamente)
        transforms.append(v2.RandomHorizontalFlip(p=0.5))
        
    return v2.Compose(transforms)

# -----------------------------
# 3. Model Loader
# -----------------------------
def get_model(num_classes):
    # Load pre-trained Faster R-CNN with FPN
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace head for custom classes
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

# -----------------------------
# 4. Training Loop
# -----------------------------
#def train_model(dataset, dataset_test, num_classes, num_epochs=10, batch_size=2, lr=0.005):
def train_model(dataset, dataset_test, num_classes, num_epochs=5, batch_size=16, lr=0.005):    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    data_loader_test = DataLoader(dataset_test, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss:.4f}")
        torch.save(model.state_dict(), "fasterrcnn_fpn.pth")

    print("Training complete.")
    return model

# -----------------------------
# 5. Run Training
# -----------------------------
if __name__ == "__main__":
    # Paths to your dataset
    
    train_images = "Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb/train/images"
    train_annotations = "Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb/train/images/_annotations.coco.json"
    
    test_images = "Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb/valid/images"
    test_annotations = "Leukocoria Detector.v2-leukocoria-version-2.yolov8-obb/valid/images/_annotations.coco.json"

    dataset = CustomDataset(train_images, train_annotations, get_transform(train=True))
    dataset_test = CustomDataset(test_images, test_annotations, get_transform(train=False))

    # num_classes = background + your object classes
    #num_classes = 1 + 3  # Example: 3 object classes
    num_classes = 1+2 # Example: 2 object classes #MOD

    
    model = train_model(dataset, dataset_test, num_classes, num_epochs=5, batch_size=16)

    # Save trained model
    torch.save(model.state_dict(), "fasterrcnn_fpn.pth")
