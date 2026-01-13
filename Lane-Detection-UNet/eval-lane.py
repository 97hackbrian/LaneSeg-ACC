import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from tqdm import tqdm
import os
from PIL import Image

# --- CONFIGURACIÓN DE GPU ---
# Usamos la GPU por defecto (0). No bloqueamos nada.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Usando dispositivo: {device}")

# --- MODELO UNET CORREGIDO ---
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)

        self.conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        # --- CORRECCIÓN AQUÍ ---
        dec2 = self.upconv2(dec3)  # Antes decía (dec2), ahora (dec3) CORRECTO
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.conv(dec1))

# --- TRANSFORMACIONES ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((720, 1280)) 
])

# --- DATASET INTELIGENTE ---
class BDD100KDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = []

        all_files = os.listdir(images_dir)
        print(f"🔎 Analizando imágenes en: {images_dir}...")

        for img_name in tqdm(all_files, desc="Verificando archivos"):
            if not img_name.endswith('.jpg'):
                continue
            
            # Probamos nombres posibles para la máscara
            candidatos = [
                img_name.replace('.jpg', '_train_id.png'),
                img_name.replace('.jpg', '_val_id.png'),
                img_name.replace('.jpg', '_id.png'),
                img_name.replace('.jpg', '.png')
            ]
            
            mask_name = None
            for c in candidatos:
                if os.path.exists(os.path.join(self.masks_dir, c)):
                    mask_name = c
                    break
            
            if mask_name:
                self.images.append((img_name, mask_name))
        
        print(f"✅ ¡Listo! Se evaluarán {len(self.images)} imágenes.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name, mask_name = self.images[idx]
        
        image_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# --- RUTAS DE ARCHIVOS ---
val_image_dir = '/home/leyla/Downloads/archive/bdd100k/bdd100k/images/10k/val'
val_mask_dir = '/home/leyla/Downloads/archive/bdd100k_seg/bdd100k/seg/labels/val'

# Cargar dataset
val_dataset = BDD100KDataset(val_image_dir, val_mask_dir, transform)

if len(val_dataset) == 0:
    print("❌ ERROR: No se encontraron imágenes. Revisa las rutas.")
    exit()

# BATCH SIZE = 1 (Para evitar CUDA Out of Memory en la 4050)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# --- CARGAR MODELO ---
model = UNet(in_channels=3, out_channels=1).to(device)
model_path = 'best_unet_lane_detection.pth'

if not os.path.exists(model_path):
    print(f"⚠️ No encontré {model_path}, buscando unet_lane_detection.pth...")
    model_path = 'unet_lane_detection.pth'

print(f"📂 Cargando pesos desde: {model_path}")

try:
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
    exit()

criterion = nn.BCEWithLogitsLoss()

# --- FUNCIÓN DE EVALUACIÓN ---
def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_f1 = 0.0
    val_jaccard = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Evaluando'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            # Convertir probabilidades a 0 o 1
            preds = outputs.cpu().numpy() > 0.5
            true = masks.cpu().numpy() > 0.5
            
            val_accuracy += accuracy_score(true.flatten(), preds.flatten())
            val_f1 += f1_score(true.flatten(), preds.flatten(), zero_division=1)
            val_jaccard += jaccard_score(true.flatten(), preds.flatten(), zero_division=1)
    
    # Promedios
    val_loss /= num_batches
    val_accuracy /= num_batches
    val_f1 /= num_batches
    val_jaccard /= num_batches
    
    print("\n" + "="*40)
    print(f"📊 RESULTADOS FINALES:")
    print(f"Validation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"Validation IoU:      {val_jaccard:.4f}  <-- Métrica Clave")
    print("="*40 + "\n")

# --- EJECUTAR ---
evaluate(model, val_loader, criterion)
