import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import os
import sys
from tqdm import tqdm

# --- CONFIGURACIÓN ---
INPUT_VIDEO_PATH = "solidWhiteRight.mp4"
OUTPUT_VIDEO_PATH = "resultado_final_yolo_unet.mp4"
UNET_MODEL_PATH = "best_unet_lane_detection.pth"

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Usando dispositivo: {device}")

# --- 1. CARGAR YOLO ---
# Detectará coches, buses, semáforos
print("🚀 Cargando YOLOv8...")
yolo_model = YOLO('yolov8n.pt') 

# --- 2. DEFINIR UNET ---
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
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
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        return torch.sigmoid(self.conv(dec1))

# --- 3. PREPARAR UNET ---
unet_model = UNet(in_channels=3, out_channels=1).to(device)
if os.path.exists(UNET_MODEL_PATH):
    print(f"📂 Cargando UNet: {UNET_MODEL_PATH}")
    checkpoint = torch.load(UNET_MODEL_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    unet_model.load_state_dict(state_dict)
    unet_model.eval()
else:
    print("⚠️ No encontré el modelo UNet, solo usaré YOLO.")
    unet_model = None

transform = transforms.Compose([
    transforms.Resize((720, 1280)),
    transforms.ToTensor()
])

# --- 4. PROCESAMIENTO COMBINADO (CORREGIDO) ---
def process_combined(frame):
    # --- A. Detección de Carriles (UNet) ---
    lane_overlay = frame.copy()
    
    if unet_model:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = unet_model(input_tensor)
            # Umbral de confianza
            mask = output.squeeze().cpu().numpy() > 0.95
        
        mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
        mask_indices = mask_resized == 1
        
        # --- CORRECCIÓN AQUÍ ---
        # Solo intentamos pintar si hay algo que pintar
        if np.any(mask_indices):
            # Crear imagen verde completa
            green_layer = np.zeros_like(frame)
            green_layer[:, :, 1] = 255
            
            # 1. Mezclar toda la imagen con verde (más seguro para OpenCV)
            blended = cv2.addWeighted(frame, 0.6, green_layer, 0.4, 0)
            
            # 2. Pegar solo las partes detectadas sobre el frame original
            lane_overlay[mask_indices] = blended[mask_indices]
        # Si no hay máscara, lane_overlay se queda igual al frame original

    # --- B. Detección de Objetos (YOLO) ---
    # conf=0.4: Confianza mínima para mostrar coches
    results = yolo_model(lane_overlay, verbose=False, conf=0.4) 
    
    # Dibujar cajas
    final_frame = results[0].plot() 
    
    return final_frame

# --- 5. EJECUCIÓN PRINCIPAL ---
def main():
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"❌ Error: No encuentro '{INPUT_VIDEO_PATH}'")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print(f"🎬 Procesando video híbrido (UNet + YOLO)...")
    
    for _ in tqdm(range(total_frames), desc="Progreso"):
        ret, frame = cap.read()
        if not ret: break
        
        try:
            processed_frame = process_combined(frame)
            out.write(processed_frame)
        except Exception as e:
            print(f"⚠️ Error en un frame: {e}")
            out.write(frame) # Si falla, guardamos el frame original para no cortar el video

    cap.release()
    out.release()
    print(f"\n✅ ¡Video Final Guardado!: {OUTPUT_VIDEO_PATH}")

if __name__ == '__main__':
    main()
