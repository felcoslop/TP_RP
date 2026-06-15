"""
Ponto de entrada do detector de malha viaria (versao final).

DUAS MANEIRAS DE RODAR (escolhidas com --modelo):

  --modelo atual       (PADRAO)  pesos publicos cityscale_vitb_512_e10.ckpt
                                 (patch 512). Melhor para screenshots claros
                                 de mapas (Google/Bing Maps).

  --modelo finetuned             pesos do fine-tune feito no Colab
                                 finetuned_spacenet_local.ckpt (patch 256).
                                 Melhor para satelite bruto/escuro e estradas
                                 de terra. Veja training/finetune_local.md.

Cada preset ja embute o tamanho de patch correto — voce nao precisa lembrar.

Exemplos:
    python run.py "imagens_teste/Captura de tela 2026-06-12 135241.png"
    python run.py "foto.png" --modelo finetuned
    python run.py "foto.png" --tta                 # robustez a rotacao (4x mais lento)
    python run.py "foto.png" --sem-preprocessamento # desliga o canny_fusion
    python run.py "foto.png" --sem-contexto         # desliga mascaras de contexto

O upscale para vias finas (estradas distantes/estreitas) e AUTOMATICO: o
scout 0.5x mede a largura tipica e, em zoom padrao, roda tambem em 2.0x.
"""

import argparse
import os
import sys

import cv2

from pipeline.preprocessing import preprocess_image
from pipeline.thick_roads import run

# Presets: nome -> (arquivo de pesos, tamanho de patch do encoder)
MODELOS = {
    "atual":     ("cityscale_vitb_512_e10.ckpt", 512),
    "finetuned": ("finetuned_spacenet_local.ckpt", 256),
}


def _resolver_pesos(nome_arquivo):
    """Procura os pesos em weights/ e depois em ../weights/."""
    for cand in (os.path.join("weights", nome_arquivo),
                 os.path.join("..", "weights", nome_arquivo)):
        if os.path.exists(cand):
            return cand
    return None


def _build_parser():
    p = argparse.ArgumentParser(
        description="Detector de malha viaria a partir de imagens de satelite.")
    p.add_argument("imagem", nargs="?", help="Caminho da imagem de entrada.")
    p.add_argument("--imagem", dest="imagem_opt", help="Caminho da imagem (alternativa).")
    p.add_argument("--modelo", default="atual", choices=tuple(MODELOS),
                   help="atual = pesos publicos 512 (padrao); "
                        "finetuned = pesos do Colab 256.")
    p.add_argument("--tta", action="store_true",
                   help="Test-time augmentation de rotacao (0/90/180/270); 4x mais lento.")
    p.add_argument("--sem-preprocessamento", action="store_true",
                   help="Desliga o realce canny_fusion (que e o padrao).")
    p.add_argument("--sem-contexto", action="store_true",
                   help="Desliga as mascaras de contexto (vegetacao/agua/telhado/solo).")
    p.add_argument("--sem-passos", action="store_true",
                   help="NAO salva as imagens de cada etapa (por padrao SALVA em passos/).")
    p.add_argument("--vias-claras", default="auto",
                   choices=("auto", "sim", "nao"),
                   help="Vias claras/neutras: 'auto' suprime so em zoom proximo/"
                        "urbano (padrao); 'nao' sempre suprime (urbano); 'sim' "
                        "nunca suprime (satelite alto / rural com terra clara).")
    p.add_argument("--brilho-min", type=int, default=205,
                   help="Limiar de brilho do portao de vias claras (0-255). "
                        "MENOR = rejeita mais claros (menos tolerante); MAIOR = "
                        "mais tolerante. Padrao 205 (quase-branco).")
    p.add_argument("--limiar-via", type=float, default=0.30,
                   help="TOLERANCIA da mascara de via (0-1). So vira rua o branco "
                        ">= este valor. MAIOR = corta linhas fracas/falsas (mais "
                        "exigente); MENOR = aceita vias mais fracas. Padrao 0.30.")
    p.add_argument("--toco-frac", type=float, default=0.80,
                   help="Remove traco interno de quarteirao (ponta solta dentro "
                        "do bloco) cujo alcance e < esta fracao do menor lado do "
                        "bloco. MAIOR = remove tocos mais longos (mas pode pegar "
                        "rua real); MENOR = mais conservador. Padrao 0.80.")
    p.add_argument("--saida", default="outputs",
                   help="Pasta de saida (padrao: outputs).")
    return p


def main():
    args = _build_parser().parse_args()

    image_path = args.imagem_opt or args.imagem
    if not image_path:
        _build_parser().print_usage()
        print("Erro: informe a imagem como argumento ou com --imagem.")
        sys.exit(1)
    if not os.path.exists(image_path):
        print(f"Erro: arquivo '{image_path}' nao encontrado.")
        sys.exit(1)

    nome_pesos, patch = MODELOS[args.modelo]
    checkpoint = _resolver_pesos(nome_pesos)
    if checkpoint is None:
        print(f"Erro: pesos '{nome_pesos}' nao encontrados em weights/.")
        print("Consulte weights/LEIA-ME.md para baixar e posicionar os arquivos.")
        if args.modelo == "finetuned":
            print("Para gerar o fine-tuned, veja training/finetune_local.md (Colab).")
        sys.exit(1)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Erro: nao foi possivel ler a imagem: {image_path}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    metodo = "none" if args.sem_preprocessamento else "canny_fusion"
    print(f"Modelo: {args.modelo} ({nome_pesos}, patch {patch})")
    print(f"Pre-processamento (so para a rede): {metodo}")
    img_for_model = preprocess_image(img_rgb, method=metodo) if metodo != "none" else None

    run(image_path, checkpoint=checkpoint, img_rgb=img_rgb,
        img_for_model=img_for_model, use_context=not args.sem_contexto,
        tta=args.tta, patch=patch, out_dir=args.saida,
        debug_steps=not args.sem_passos, vias_claras=args.vias_claras,
        bright_vmin=args.brilho_min, road_thr=args.limiar_via,
        toco_frac=args.toco_frac)


if __name__ == "__main__":
    main()
