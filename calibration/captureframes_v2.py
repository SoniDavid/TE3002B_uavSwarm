import cv2
import os
from djitellopy import Tello

# Configuraciones
SAVE_DIR = "../calib_frames/calib_framesAbraham"
TELLO_IP = "192.168.0.100"  # Definimos la IP estática aquí

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Instanciamos el Tello con la IP directa
    tello = Tello(host=TELLO_IP)
    count = 0

    try:
        # 2. Conectamos al dron
        tello.connect()
        battery_info = tello.get_battery()
        print(f"Drone battery soc: {battery_info}%")

        # 3. Encendemos el stream de video
        tello.streamon()
        frame_read = tello.get_frame_read()

        print("\n=== CONTROLES DE CÁMARA ===")
        print("'s' = guardar frame  |  'q' = salir")
        print("===========================\n")

        # 4. Bucle principal de captura
        while True:
            frame = frame_read.frame
            
            # Pequeña protección por si el buffer de video aún no tiene imágenes
            if frame is None:
                continue

            display = frame.copy()
            cv2.putText(display, f"Frames: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Tello", display)

            key = cv2.waitKey(1) & 0xFF
            
            # Lógica de teclado
            if key == ord('s'):
                path = os.path.join(SAVE_DIR, f"frame_{count:03d}.png")
                cv2.imwrite(path, frame)
                print(f"  [+] Guardado: {path}")
                count += 1
            elif key == ord('q'):
                print("Saliendo del bucle de captura...")
                break

    finally:
        # 5. Cierre seguro (Se ejecuta SIEMPRE, haya errores o no)
        print(f"\nTotal capturado: {count} frames")
        print("Cerrando stream y limpiando recursos...")
        
        try:
            tello.streamoff()
        except Exception:
            pass
            
        tello.end()
        cv2.destroyAllWindows()