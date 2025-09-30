from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PIL import Image
import matplotlib.pyplot as plt

# Función para recortar con margen
def create_detection_padding(imagen: np.ndarray, caja: np.ndarray, margen: int = 5) -> np.ndarray:
    x1, y1, x2, y2 = caja.astype(int)
    alto, ancho, _ = imagen.shape
    x1 = max(0, x1 - margen)
    y1 = max(0, y1 - margen)
    x2 = min(ancho, x2 + margen)
    y2 = min(alto, y2 + margen)
    return imagen[y1:y2, x1:x2]

# main function, que detecta, luego recorta y clasifica sobre el recorte
def procesar_imagen(ruta_imagen: Path, margen: int = 5):
    # Abrir la imagen
    imagen = np.array(Image.open(ruta_imagen).convert("RGB"))

    # Cargar detector y clasificador preseleccionados
    detector = pw_detection.MegaDetectorV6(version="MDV6-yolov10-c") #Aquí se seleccionó un modelo en partícular
    clasificador = pw_classification.AI4GAmazonRainforest()

    # Detectar
    resultados = detector.predictor(imagen)  # lista de detecciones

    # Entonces, nos fijamos en cada una de las detecciones
    for i, result in enumerate(resultados):
        cajas = result.boxes.xyxy.cpu().numpy()  # coordenadas como vector de numpy [x1, y1, x2, y2]

        for j, caja in enumerate(cajas):
            recorte = create_detection_padding(result.orig_img, caja, margen)
            especie_predicha = clasificador.single_image_classification(recorte)

            # Solo el nombre de la especie con mayor confianza
            nombre_especie = especie_predicha['prediction']

            print(f"Detección {i+1}.{j+1}: Caja = {caja}, Especie predicha = {nombre_especie}")

            # Guardaremos el recorte con nombre válido
            nombre_recorte = ruta_imagen.parent / f"{ruta_imagen.stem}_det{i+1}_{j+1}_{nombre_especie}.jpg"
            Image.fromarray(recorte).save(nombre_recorte)
            print(f"Recorte guardado en: {nombre_recorte}")


# Bloque de ejecución desde CMD
if __name__ == "__main__":
    parser = ArgumentParser(
        prog='plantilla_ejemplo',
        description="Script para detectar y clasificar animales en una imagen"
    )
    parser.add_argument("ruta_imagen", type=Path, help="Ruta de la imagen")
    parser.add_argument("--margen", type=int, default=5, help="Tamaño del margen alrededor de la detección")

    args = parser.parse_args()
    procesar_imagen(args.ruta_imagen, args.margen)
