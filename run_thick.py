import argparse
import os
import sys

import cv2

from pipeline.preprocessing import preprocess_image
from pipeline.thick_roads import run

PREPROCESS_OPTIONS = (
    "none",
    "gaussian",
    "clahe",
    "sharpen",
    "sobel_fusion",
    "canny_fusion",
)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Pipeline Thick/Tiny para deteccao de vias."
    )
    parser.add_argument("image", nargs="?", help="Caminho da imagem de entrada.")
    parser.add_argument("--image", dest="image_opt", help="Caminho da imagem de entrada.")
    parser.add_argument(
        "--preprocess",
        default="none",
        choices=PREPROCESS_OPTIONS,
        type=str.lower,
        help="Metodo de pre-processamento classico.",
    )
    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Salva a imagem pre-processada em outputs/.",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image_path = args.image_opt or args.image
    if not image_path:
        parser.print_usage()
        print("Erro: informe a imagem com --image ou como argumento posicional.")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Erro: Arquivo '{image_path}' nao encontrado.")
        sys.exit(1)

    # Busca dinamica do arquivo de pesos
    checkpoint = "weights/cityscale_vitb_512_e10.ckpt"
    if not os.path.exists(checkpoint):
        fallback = "../weights/cityscale_vitb_512_e10.ckpt"
        if os.path.exists(fallback):
            checkpoint = fallback
        else:
            print("Erro: Checkpoint 'cityscale_vitb_512_e10.ckpt' nao encontrado.")
            print("Por favor, crie a pasta 'weights/' e coloque o arquivo nela.")
            print("Consulte 'documentacao_pesos.md' para os links de download e instrucoes.")
            sys.exit(1)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Erro: Imagem nao encontrada: {image_path}")
        sys.exit(1)

    img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"Pre-processamento: {args.preprocess}")
    img_rgb = preprocess_image(img_rgb_original, method=args.preprocess)

    if args.save_preprocessed:
        os.makedirs("outputs", exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"preprocessed_{base}_{args.preprocess}.png"
        output_path = os.path.join("outputs", filename)
        cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"Imagem pre-processada salva em: {output_path}")

    print(f"Usando checkpoint: {checkpoint}")
    run(image_path, checkpoint=checkpoint, img_rgb=img_rgb)

if __name__ == "__main__":
    main()
