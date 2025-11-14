from pathlib import Path 
import numpy as np 
from PIL import Image 


def crop_with_margin(image: np.ndarray, box: np.ndarray, margin: int = 5) -> np.ndarray:
    """ Recorta imágenes por un recuadro aumentando un margen

    Args:
        image (np.ndarray): array de la imagen
        box (np.ndarray[int]): coordenadas de la caja para el recorte
        margin (int): tamaño en pixeles del margen alredor de la caja

    Returns:
        np.ndarray: array de la imagen recortada
    """
    x1, y1, x2, y2 = box.astype(int)
    height, width, _ = image.shape
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(width, x2 + margin)
    y2 = min(height, y2 + margin)
    return image[y1:y2, x1:x2]


#Esta función es la que procesará la imagen, es decir,
#Carga la imagen, hace la detección (o las detecciones)
#Recorta las detecciones 
#Claficia individualmente 
def process_image(image_path: Path, output_folder: Path, detector, classifier, margin: int = 5):
    image = np.array(Image.open(image_path).convert("RGB"))
    results = detector.predictor(image)

#Para cada una de las detecciones 
    for i, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu().numpy()
	#Recorta cada ua de las detecciónes
        for j, box in enumerate(boxes):
            cropped = crop_with_margin(result.orig_img, box, margin)
            predicted_species = classifier.single_image_classification(cropped)
#Predice y clasifica cada una de las imagenes 
            species_name = predicted_species['prediction']

#Le da formato a la imagen resultante 
            output_name = output_folder / f"{image_path.stem}_det{i+1}_{j+1}_{species_name}.jpg"
            Image.fromarray(cropped).save(output_name)


#Esta función hace lo de arriba pero con un folder entero
def process_folder(folder_path: Path, detector, classifier, margin: int = 5):
    output_folder = folder_path / "recortes"
    output_folder.mkdir(exist_ok=True)

#Aplica la función anterior a las imagenes del folder
    for image_path in sorted(folder_path.glob("*.jpg")):
        process_image(image_path, output_folder, detector, classifier, margin)
