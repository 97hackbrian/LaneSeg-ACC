import torch
import onnx
import sys
import os

sys.path.append(os.getcwd())

try:
    from models.unet import UNet
except ImportError as e:
    print("Error importando UNet. Verifica 'models/unet.py'")
    sys.exit(1)

# --- CONFIGURACIÓN ---
pth_path = "unet_best.pt"
onnx_path = "lane_unet.onnx"
input_shape = (1, 3, 256, 256)

print(f"--> Inicializando UNet (in=3, out=1)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1)

print(f"--> Cargando checkpoint desde {pth_path}...")
try:
    # 1. Cargar el archivo completo (es un diccionario)
    checkpoint = torch.load(pth_path, map_location=device)
    
    # 2. Verificar si es un checkpoint completo o solo pesos
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("    Detectado formato Checkpoint completo.")
        print(f"    (Info: Epoch {checkpoint.get('epoch', '?')}, Loss {checkpoint.get('val_loss', '?')})")
        weights = checkpoint['model_state_dict']
    else:
        print("    Detectado formato State Dict simple.")
        weights = checkpoint

    # 3. Cargar los pesos al modelo
    model.load_state_dict(weights)
    print("    ¡Pesos cargados exitosamente!")

except Exception as e:
    print(f"\nERROR FATAL AL CARGAR PESOS:\n{e}")
    sys.exit(1)

model.to(device)
model.eval()

# --- EXPORTAR A ONNX ---
dummy_input = torch.randn(*input_shape).to(device)

print(f"--> Exportando a {onnx_path}...")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=11,
    input_names=['input_0'],
    output_names=['output_0'],
    do_constant_folding=True
)

print(f"\n¡LISTO! Modelo guardado en: {onnx_path}")
print(f"Usa estos nombres en Isaac ROS -> Input: 'input_0', Output: 'output_0'")
