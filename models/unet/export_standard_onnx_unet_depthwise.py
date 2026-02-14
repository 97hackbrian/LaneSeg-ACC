import torch
import onnx
import sys
import os

sys.path.append(os.getcwd())

try:
    from models.unet_depthwise import UNetDepthwise
except ImportError as e:
    print("Error importando UNetDepthwise. Verifica 'models/unet_depthwise.py'")
    sys.exit(1)

# --- CONFIGURACIÓN ---
pth_path = "unet_depthwise_best.pt"
onnx_path = "lane_unet_depthwise.onnx"
# Dimensiones estáticas (Batch=1)
input_shape = (1, 3, 256, 256)

print(f"--> Inicializando UNet...")
device = torch.device("cpu") # Usamos CPU para exportar para evitar ruido de CUDA
model = UNetDepthwise(in_channels=3, out_channels=1)

print(f"--> Cargando pesos...")
try:
    checkpoint = torch.load(pth_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
except Exception as e:
    print(f"Error cargando: {e}")
    sys.exit(1)

model.eval()

# --- EXPORTAR A ONNX (ESTÁTICO) ---
dummy_input = torch.randn(*input_shape).to(device)

print(f"--> Exportando a {onnx_path}...")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=11,
    input_names=['input_0'],
    output_names=['output_0'],
    # Desactivar ejes dinámicos para máxima estabilidad en Isaac ROS 2.1
    dynamic_axes=None, 
    do_constant_folding=True
)

print(f"\n¡ÉXITO! Modelo guardado. Formato NCHW estándar.")