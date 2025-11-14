from pathlib import Path
from argparse import ArgumentParser
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification 
from utils import process_image, process_folder 

# Detectores probados basados en los disponibles en https://microsoft.github.io/CameraTraps/model_zoo/megadetector/
DETECTORS = {
    "YOLOv9_Compact": "MDV6-yolov9-c",
    "YOLOv9_Extra": "MDV6-yolov9-e",
    "YOLOv10_Compact": "MDV6-yolov10-c",
    "YOLOv10_Extra": "MDV6-yolov10-e",
    "RtDetr_Compact": "MDV6-rtdetr-c"
}
DETECTORS_OPTS = list(DETECTORS.keys())

# Clasificadores probados de la lista disponible en https://microsoft.github.io/CameraTraps/model_zoo/classifiers/
CLASSIFIERS = {
    "AI4G Amazon Rainforest": "AI4GAmazonRainforest",
    "AI4G Snapshot Serengeti": "AI4GSnapshotSerengeti",
    "AI4G Opossum": "AI4GOpossum",
    "AI4G Central Africa": "AI4GCentralAfrica"
}
CLASSIFIERS_OPTS = list(CLASSIFIERS.keys())

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="main_detector_classifier",
        description="Detecta y clasifica animales en imágenes o carpetas, permitiendo elegir modelos."
    )
    parser.add_argument("path", type=Path, help="Ruta de la imagen o carpeta a procesar")
    parser.add_argument("--detector", default=DETECTORS_OPTS[0], choices=DETECTORS_OPTS, 
                        help="Modelo de detección que se desea usará")
    parser.add_argument("--classifier", default=CLASSIFIERS_OPTS[0], choices=CLASSIFIERS_OPTS,
                        help="Modelo de clasificación que se desea usará")
    parser.add_argument("--margin", type=int, default=5, help="Tamaño del margen alrededor de la detección")

    args = parser.parse_args()

    # Cargaremos los modelos y se imprimirá la elección
    print(f"\nCargando detector: {args.detector}")
    detector = pw_detection.MegaDetectorV6(version=DETECTORS[args.detector])

    print(f"Cargando clasificador: {args.classifier}")
    classifier = getattr(pw_classification, CLASSIFIERS[args.classifier])()

    # Si path es directorio se clasifican todos las imagenes en el directorio
    if args.path.is_dir():
        process_folder(args.path, detector, classifier, args.margin)
    elif args.path.is_file():
        output_folder = args.path.parent / "recortes_single"
        output_folder.mkdir(exist_ok=True)
        process_image(args.path, output_folder, detector, classifier, args.margin)
    else:
        raise ValueError(f"{args.path} no es un valor válido")
