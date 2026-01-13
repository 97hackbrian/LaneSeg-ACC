import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import sys
from tqdm import tqdm

# --- CONFIGURACIÓN ESPECÍFICA ---
INPUT_VIDEO_PATH = "solidWhiteRight.mp4"  # <--- YA CONFIGURADO
OUTPUT_VIDEO_PATH = "resultado_video_lane.mp4"
MODEL_PATH = "best_unet_lane_detection.pth"

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Usando dispositivo: {device}")

# --- CLASE UNET (CORREGIDA) ---
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
        
        # Corrección aplicada
        dec2 = self.upconv2(dec3) 
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        return torch.sigmoid(self.conv(dec1))

# --- PROCESAMIENTO DE FRAME ---
def process_frame(frame, model, transform):
    # Convertir BGR (OpenCV) a RGB (PIL)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Transformar y enviar a GPU
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # Inferencia
    with torch.no_grad():
        output = model(input_tensor)
        mask = output.squeeze().cpu().numpy() > 0.5 
    
    # Redimensionar máscara al tamaño original del frame
    mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
    
    # Crear capa verde
    green_overlay = np.zeros_like(frame)
    green_overlay[:, :, 1] = 255 # Canal verde
    
    # Mezclar
    alpha = 0.5
    mask_indices = mask_resized == 1
    
    result = frame.copy()
    result[mask_indices] = cv2.addWeighted(
        frame[mask_indices], 1 - alpha, 
        green_overlay[mask_indices], alpha, 
        0
    )
    
    return result

# --- MAIN ---
def main():
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"❌ ERROR: No encuentro el video '{INPUT_VIDEO_PATH}' en esta carpeta.")
        return

    # Cargar Modelo
    print(f"📂 Cargando modelo: {MODEL_PATH}")
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Transformaciones (720x1280 para UNet)
    transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor()
    ])

    # Abrir Video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎞️ Procesando '{INPUT_VIDEO_PATH}': {width}x{height} a {fps} FPS ({total_frames} frames)")

    # Configurar salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    # Bucle
    for _ in tqdm(range(total_frames), desc="Renderizando"):
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = process_frame(frame, model, transform)
        out.write(result_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ ¡Terminado! Video guardado como: {OUTPUT_VIDEO_PATH}")

if __name__ == '__main__':
    main()
