#Este código hace lo mismo que plantilla_ejemplo2
#Pero usará otro detector y clasificador
# Librerías necesarias

from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PIL import Image

# Detector y clasificador diferentes
detector = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")
clasificador = pw_classification.AI4GOpossum()

# Tomamos en cuenta la misma función que recorta imagenes 
def recortar_con_margen(imagen: np.ndarray, caja: np.ndarray, margen: int = 5) -> np.ndarray:
    x1, y1, x2, y2 = caja.astype(int)
    alto, ancho, _ = imagen.shape
    x1 = max(0, x1 - margen)
    y1 = max(0, y1 - margen)
    x2 = min(ancho, x2 + margen)
    y2 = min(alto, y2 + margen)
    return imagen[y1:y2, x1:x2]

# La función principal, que es la misma del script anterior.

def procesar_imagen(ruta_imagen: Path, carpeta_salida: Path, margen: int = 5):
    imagen = np.array(Image.open(ruta_imagen).convert("RGB"))

    resultados = detector.predictor(imagen)

    for i, result in enumerate(resultados):
        cajas = result.boxes.xyxy.cpu().numpy()
        for j, caja in enumerate(cajas):
            recorte = recortar_con_margen(result.orig_img, caja, margen)
            especie_predicha = clasificador.single_image_classification(recorte)
            nombre_especie = especie_predicha['prediction']

            print(f"Imagen: {ruta_imagen.name} | Detección {i+1}.{j+1}: Caja = {caja}, Especie = {nombre_especie}")

            # Guardar recorte en la carpeta de salida diferente (recortes2)
            nombre_recorte = carpeta_salida / f"{ruta_imagen.stem}_det{i+1}_{j+1}_{nombre_especie}.jpg"
            Image.fromarray(recorte).save(nombre_recorte)
            print(f"Recorte guardado en: {nombre_recorte}")

# Esta función servirá para procesar todas las imágenes de una carpeta
def procesar_carpeta(carpeta: Path, margen: int = 5):
    carpeta_salida = carpeta / "recortes3"
    carpeta_salida.mkdir(exist_ok=True)  # crea la carpeta si no existe

    for ruta_imagen in sorted(carpeta.glob("foto*.jpg")):
        procesar_imagen(ruta_imagen, carpeta_salida, margen)

# Ejecución
if __name__ == "__main__":
    parser = ArgumentParser(
        prog='plantilla_ejemplo',
        description="Detecta y clasifica animales en todas las imágenes de una carpeta y guarda los recortes en subcarpeta"
    )
    parser.add_argument("ruta_carpeta", type=Path, help="Ruta de la carpeta con imágenes")
    parser.add_argument("--margen", type=int, default=5, help="Tamaño del margen alrededor de la detección")

    args = parser.parse_args()
    procesar_carpeta(args.ruta_carpeta, args.margen)
