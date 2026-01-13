import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- 1. CONFIGURACIÓN ---
# Detectar GPU automáticamente
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Usando dispositivo: {device}")

# Rutas (Ajustadas a tu entorno)
MODEL_PATH = 'best_unet_lane_detection.pth'
OUTPUT_FILENAME = 'inference_result.png'

# --- 2. CLASE UNET (LA VERSIÓN CORREGIDA) ---
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
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3) # <--- CORREGIDO (dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.conv(dec1))

# --- 3. CARGAR EL MODELO ---
model = UNet(in_channels=3, out_channels=1).to(device)

if not os.path.exists(MODEL_PATH):
    print(f"⚠️ No encontré '{MODEL_PATH}'. Intentando con 'unet_lane_detection.pth'...")
    MODEL_PATH = 'unet_lane_detection.pth'

print(f"📂 Cargando modelo: {MODEL_PATH}")

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
except Exception as e:
    print(f"❌ Error crítico cargando el modelo: {e}")
    sys.exit(1)

# --- 4. PROCESAMIENTO E INFERENCIA ---
def run_inference(image_path):
    print(f"🖼️ Procesando imagen: {image_path}")
    
    # 1. Cargar imagen original
    try:
        original_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ No se pudo abrir la imagen: {e}")
        return

    # 2. Transformar para el modelo (720x1280)
    transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # 3. Predicción
    with torch.no_grad():
        output = model(input_tensor)
        # La salida ya tiene sigmoid aplicado en forward, es un valor 0-1
        pred_mask = output.squeeze().cpu().numpy() > 0.5

    # 4. Visualización (Overlay)
    # Redimensionamos la imagen original a 1280x720 para que coincida con la máscara
    img_np = np.array(original_img.resize((1280, 720)))
    
    # Crear máscara verde
    # Creamos una imagen vacía del mismo tamaño
    green_layer = np.zeros_like(img_np)
    # Llenamos el canal G (verde) con 255
    green_layer[:, :, 1] = 255 

    # Mezclar imagen original con capa verde solo donde pred_mask es True
    alpha = 0.6 # Opacidad del carril (0.0 transparente - 1.0 sólido)
    
    output_img = img_np.copy()
    output_img[pred_mask] = (
        output_img[pred_mask] * (1 - alpha) + 
        green_layer[pred_mask] * alpha
    ).astype(np.uint8)

    # 5. Guardar resultado comparativo
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Imagen Original")
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Detección de Carril (UNet)")
    plt.imshow(output_img)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME)
    print(f"✅ ¡Éxito! Resultado guardado en: {os.path.abspath(OUTPUT_FILENAME)}")
    plt.close()

# --- 5. EJECUCIÓN AUTOMÁTICA ---
if __name__ == "__main__":
    # Buscamos una imagen de prueba automáticamente en tu carpeta de validación
    # Si quieres probar otra, cambia esta ruta manualmente.
    val_images_path = '/home/leyla/Downloads/archive/bdd100k/bdd100k/images/10k/val'
    
    if os.path.exists(val_images_path):
        all_images = [f for f in os.listdir(val_images_path) if f.endswith('.jpg')]
        if all_images:
            # Tomamos la primera imagen de la lista
            test_image = os.path.join(val_images_path, all_images[0])
            run_inference(test_image)
        else:
            print("⚠️ La carpeta existe pero no tiene imágenes .jpg")
    else:
        print(f"⚠️ No encontré la carpeta de validación: {val_images_path}")
        print("Por favor, edita la variable 'val_images_path' en el script.")
