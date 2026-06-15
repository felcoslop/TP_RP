"""
Processa TODAS as imagens de uma pasta de uma vez (carregando o modelo uma
unica vez) e salva, para cada imagem, o resultado completo + as imagens de
CADA passo do pipeline.

Uso (a partir de versão_2/):
    python processar_pasta.py                       # processa imagens_teste/
    python processar_pasta.py minha_pasta           # outra pasta
    python processar_pasta.py imagens_teste --modelo finetuned --limiar-via 0.45

Saidas: outputs/<nome_da_imagem>/ com thick_overlay.png, thick_surface.png,
thick_road_mask.png, context_masks.png, graph_output.json/.graphml e a
subpasta passos/ (1_original ... 9_overlay_final).
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import MODELOS, _resolver_pesos
from pipeline.preprocessing import preprocess_image
from pipeline.thick_roads import ThickRoadDetector, run_pipeline
from pipeline.export import save_graph_json, save_graph_graphml
from pipeline.graph_refine import COLORS_RGB

_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _save_all(result, image_path, out_dir):
    """Grava todas as saidas + a subpasta passos/ (mesmos arquivos do run.py)."""
    os.makedirs(out_dir, exist_ok=True)
    overlay = result["overlay"]
    surface = result["surface_map"]

    cv2.imwrite(os.path.join(out_dir, "thick_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "thick_road_mask.png"),
                (result["fused_road"] * 255).clip(0, 255).astype(np.uint8))
    sv = np.zeros((*surface.shape, 3), np.uint8)
    sv[surface == 1] = COLORS_RGB[1]
    sv[surface == 2] = COLORS_RGB[2]
    cv2.imwrite(os.path.join(out_dir, "thick_surface.png"),
                cv2.cvtColor(sv, cv2.COLOR_RGB2BGR))
    if result.get("masks_viz") is not None:
        cv2.imwrite(os.path.join(out_dir, "context_masks.png"),
                    cv2.cvtColor(result["masks_viz"], cv2.COLOR_RGB2BGR))
    if result.get("graph") is not None:
        save_graph_json(result["graph"], os.path.join(out_dir, "graph_output.json"),
                        source_image=os.path.basename(image_path))
        save_graph_graphml(result["graph"], os.path.join(out_dir, "graph_output.graphml"))
    if result.get("steps"):
        pd = os.path.join(out_dir, "passos")
        os.makedirs(pd, exist_ok=True)
        for nome, im in result["steps"].items():
            cv2.imwrite(os.path.join(pd, f"{nome}.png"),
                        cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta", nargs="?", default="imagens_teste",
                    help="Pasta com as imagens (padrao: imagens_teste).")
    ap.add_argument("--modelo", default="atual", choices=tuple(MODELOS))
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--sem-contexto", action="store_true")
    ap.add_argument("--sem-preprocessamento", action="store_true")
    ap.add_argument("--limiar-via", type=float, default=0.30)
    ap.add_argument("--toco-frac", type=float, default=0.80)
    ap.add_argument("--vias-claras", default="auto", choices=("auto", "sim", "nao"))
    ap.add_argument("--brilho-min", type=int, default=205)
    ap.add_argument("--saida", default="outputs")
    args = ap.parse_args()

    if not os.path.isdir(args.pasta):
        print(f"Erro: pasta '{args.pasta}' nao encontrada.")
        sys.exit(1)

    nome_pesos, patch = MODELOS[args.modelo]
    ckpt = _resolver_pesos(nome_pesos)
    if ckpt is None:
        print(f"Erro: pesos '{nome_pesos}' nao encontrados em weights/.")
        sys.exit(1)

    imgs = sorted(f for f in os.listdir(args.pasta)
                  if f.lower().endswith(_EXTS))
    if not imgs:
        print(f"Nenhuma imagem em '{args.pasta}'.")
        sys.exit(1)

    print(f"Modelo: {args.modelo} ({nome_pesos}, patch {patch}) | {len(imgs)} imagens")
    detector = ThickRoadDetector(ckpt, patch=patch)
    metodo = "none" if args.sem_preprocessamento else "canny_fusion"

    for i, f in enumerate(imgs, 1):
        path = os.path.join(args.pasta, f)
        name = os.path.splitext(f)[0]
        print(f"\n===== [{i}/{len(imgs)}] {f} =====")
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print("  (ignorada: nao foi possivel ler)")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ifm = preprocess_image(img_rgb, metodo) if metodo != "none" else None
        result = run_pipeline(
            img_rgb, detector, use_context=not args.sem_contexto, tta=args.tta,
            img_for_model=ifm, debug_steps=True, vias_claras=args.vias_claras,
            bright_vmin=args.brilho_min, road_thr=args.limiar_via,
            toco_frac=args.toco_frac)
        out_dir = os.path.join(args.saida, name)
        _save_all(result, path, out_dir)
        print(f"  -> {out_dir}/ (overlay, surface, mascaras, grafo, passos/)")

    print("\nFIM — todas as imagens processadas.")


if __name__ == "__main__":
    main()
