import sys
import time

def check_installation():
    print("--- 1. Verificando Importación ---")
    try:
        import cupy as cp
        import numpy as np
        print(f"✅ CuPy importado correctamente.")
        print(f"   Versión CuPy: {cp.__version__}")
    except ImportError as e:
        print(f"❌ FALLO: No se encuentra CuPy. ({e})")
        return

    print("\n--- 2. Verificando Hardware y Drivers ---")
    try:
        # Obtener info del dispositivo 0
        dev = cp.cuda.Device(0)
        dev.use()
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props['name'].decode('utf-8')
        
        # Versiones de CUDA
        driver_ver = cp.cuda.runtime.driverGetVersion() # Lo que soporta tu driver
        runtime_ver = cp.cuda.runtime.runtimeGetVersion() # Con lo que se compiló CuPy
        
        print(f"✅ GPU Detectada: {name}")
        print(f"   CUDA Driver Version: {driver_ver}")
        print(f"   CUDA Runtime Version (CuPy): {runtime_ver}")
        
        # Validación de compatibilidad básica
        if runtime_ver > driver_ver:
             print("⚠️  ADVERTENCIA: La versión de CuPy (Runtime) es mayor que la de tu Driver.")
             print("    Esto podría causar errores. Considera actualizar drivers o bajar la versión de CuPy.")
        else:
             print("✅ Compatibilidad Driver/Runtime: OK")
             
    except Exception as e:
        print(f"❌ FALLO al acceder a la GPU: {e}")
        print("   ¿Tienes el runtime de nvidia activado? (nvidia-container-runtime)")
        return

    print("\n--- 3. Test de Funcionalidad (Memoria y Cálculo) ---")
    try:
        # Crear array en GPU
        a_gpu = cp.array([1, 2, 3, 4, 5], dtype=np.float32)
        # Operación matemática en GPU
        b_gpu = cp.sin(a_gpu) * 10
        # Mover resultado a CPU para verificar
        b_cpu = cp.asnumpy(b_gpu)
        
        print(f"✅ Operación matemática en GPU exitosa.")
        print(f"   Resultado muestra: {b_cpu[:2]}...")
    except Exception as e:
        print(f"❌ FALLO durante el cálculo: {e}")

if __name__ == "__main__":
    check_installation()