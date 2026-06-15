"""
Gera as figuras do relatorio (latex/figuras/) e da apresentacao
(latex/apresentacao/figuras/) a partir das saidas em outputs/.

Produz:
  - grid_<data>.png  : os 9 passos de cada imagem (4 em cima, 5 embaixo),
                       rotulados, com o nome da imagem no topo (apresentacao).
  - dataset_exemplo.png : tile do SpaceNet — imagem ORIGINAL primeiro, depois a
                          mascara e o grafo de referencia (relatorio+apresentacao).
  - passo_1..9.png   : os 9 passos de uma imagem representativa (relatorio).
  - resultado_<data>.png : overlay final de cada imagem (relatorio).

Uso (a partir de versão_2/):  python montar_figuras.py
"""

import os
import cv2
import numpy as np

V2 = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(V2, "outputs")
FIG_REL = os.path.join(V2, "latex", "figuras")
FIG_APR = os.path.join(V2, "latex", "apresentacao", "figuras")
DATA = os.path.join(V2, "..", "data")
os.makedirs(FIG_REL, exist_ok=True)
os.makedirs(FIG_APR, exist_ok=True)


def imread_u(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread Unicode-safe (caminhos com 'ã' quebram o imread no Windows)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags) if data.size else None
    except Exception:
        return None


def imwrite_u(path, img, maxw=1000):
    """cv2.imwrite Unicode-safe, com limite de largura (PDF leve)."""
    if maxw and img.shape[1] > maxw:
        s = maxw / img.shape[1]
        img = cv2.resize(img, (maxw, int(img.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok

STEPS = ["1_original", "2_preprocessamento_canny", "3_probabilidade_modelo",
         "4_mascaras_contexto", "5_probabilidade_suprimida", "6_binaria",
         "7_grafo_bruto", "8_grafo_refinado", "9_overlay_final"]
LABELS = ["1. Original", "2. Pre-proc (canny)", "3. Prob. modelo",
          "4. Mascaras contexto", "5. Prob. suprimida", "6. Binaria",
          "7. Grafo bruto", "8. Grafo refinado", "9. Overlay final"]


def _sanitize(folder):
    return folder.replace("Captura de tela ", "").replace(" ", "_")


def _labeled_thumb(img, label, tw):
    h, w = img.shape[:2]
    th = max(1, int(tw * h / w))
    im = cv2.resize(img, (tw, th))
    bar = np.zeros((36, tw, 3), np.uint8)
    cv2.putText(bar, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.64,
                (60, 220, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, im])


def build_step_grid(folder, tw=460):
    """9 passos: 4 na linha de cima, 5 na de baixo. Rotulados + titulo."""
    pd = os.path.join(OUT, folder, "passos")
    thumbs = []
    for s, lab in zip(STEPS, LABELS):
        im = imread_u(os.path.join(pd, s + ".png"))
        if im is None:
            im = np.full((260, 400, 3), 40, np.uint8)
        thumbs.append(_labeled_thumb(im, lab, tw))
    th = max(t.shape[0] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0,
              cv2.BORDER_CONSTANT, value=(20, 20, 20)) for t in thumbs]
    gap = np.full((th, 8, 3), 20, np.uint8)

    def _row(items):
        out = items[0]
        for it in items[1:]:
            out = np.hstack([out, gap, it])
        return out

    row1 = _row(thumbs[0:4])      # 4 imagens
    row2 = _row(thumbs[4:9])      # 5 imagens
    W = max(row1.shape[1], row2.shape[1])
    def _padw(r):
        if r.shape[1] < W:
            pad = W - r.shape[1]
            return cv2.copyMakeBorder(r, 0, 0, pad // 2, pad - pad // 2,
                                      cv2.BORDER_CONSTANT, value=(20, 20, 20))
        return r
    row1, row2 = _padw(row1), _padw(row2)
    vgap = np.full((10, W, 3), 20, np.uint8)
    title = np.full((46, W, 3), 20, np.uint8)
    cv2.putText(title, folder, (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    grid = np.vstack([title, row1, vgap, row2])
    out = os.path.join(FIG_APR, f"grid_{_sanitize(folder)}.png")
    imwrite_u(out, grid, maxw=2200)   # composto de 9 paineis: maior p/ ler rotulos
    return out


def build_dataset_example():
    """Tile do SpaceNet: ORIGINAL primeiro, depois mascara GT, grafo e overlay."""
    tile = "AOI_2_Vegas_116"
    rgb = imread_u(os.path.join(DATA, tile + "__rgb.png"))
    gt = imread_u(os.path.join(DATA, tile + "__gt.png"))
    if rgb is None:
        print("dataset: tile nao encontrado, pulando")
        return
    
    H_img, W_img = rgb.shape[:2]
    p_path = os.path.join(DATA, tile + "__gt_graph_dense.p")
    
    # Try drawing from pickle for perfect alignment
    if os.path.exists(p_path):
        import pickle
        gd = np.zeros((H_img, W_img, 3), dtype=np.uint8)
        try:
            with open(p_path, 'rb') as f:
                g = pickle.load(f)
            # Draw edges
            for node, neighbors in g.items():
                y, x = node
                col = int(x)
                row = int(H_img - 1 - y)
                for nb in neighbors:
                    ny, nx = nb
                    ncol = int(nx)
                    nrow = int(H_img - 1 - ny)
                    cv2.line(gd, (col, row), (ncol, nrow), (255, 255, 255), 2)
            # Draw nodes
            for node in g.keys():
                y, x = node
                col = int(x)
                row = int(H_img - 1 - y)
                cv2.circle(gd, (col, row), 4, (255, 255, 255), -1)
        except Exception as e:
            print(f"Error drawing graph from pickle: {e}")
            gd = imread_u(os.path.join(DATA, tile + "__gt_graph_dense.png"))
    else:
        gd = imread_u(os.path.join(DATA, tile + "__gt_graph_dense.png"))

    H = 460

    def lab(im, t):
        if im is None:
            im = np.full((H, H, 3), 40, np.uint8)
        im = cv2.resize(im, (H, H))
        cv2.rectangle(im, (0, 0), (H, 34), (0, 0, 0), -1)
        cv2.putText(im, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return im
        
    if gd is None:
        gd = np.full((H_img, W_img, 3), 40, np.uint8)
        
    gd_resized = cv2.resize(gd, (rgb.shape[1], rgb.shape[0]))
    overlay = rgb.copy()
    overlay[gd_resized[:, :, 0] > 127] = [0, 0, 255] # Vermelho em BGR
    
    gap = np.full((H, 8, 3), 255, np.uint8)
    montage = np.hstack([lab(rgb, "1) Imagem original (entrada)"), gap,
                         lab(gt, "2) Mascara de via (GT)"), gap,
                         lab(gd, "3) Grafo de referencia"), gap,
                         lab(overlay, "4) Grafo sobreposto")])
    for d in (FIG_REL, FIG_APR):
        imwrite_u(os.path.join(d, "dataset_exemplo.png"), montage)



def build_report_steps(folder="Captura de tela 2026-06-12 135241"):
    """Passos individuais de uma imagem representativa (relatorio: passo_N.png;
    apresentacao: nomes originais usados nos slides de explicacao do pipeline)."""
    pd = os.path.join(OUT, folder, "passos")
    for i, s in enumerate(STEPS, 1):
        im = imread_u(os.path.join(pd, s + ".png"))
        if im is None:
            continue
        imwrite_u(os.path.join(FIG_REL, f"passo_{i}.png"), im)
        imwrite_u(os.path.join(FIG_APR, s + ".png"), im)  # refresca a apresentacao
    # tambem a mascara de contexto como exemplo da legenda de cores
    ctx = imread_u(os.path.join(OUT, "Captura de tela 2026-05-13 020059",
                                  "context_masks.png"))
    if ctx is not None:
        imwrite_u(os.path.join(FIG_REL, "contexto_legenda.png"), ctx)


def build_result_overlays():
    for folder in sorted(os.listdir(OUT)):
        if not os.path.isdir(os.path.join(OUT, folder)):
            continue
        ov = os.path.join(OUT, folder, "thick_overlay.png")
        im = imread_u(ov) if os.path.isfile(ov) else None
        if im is not None:
            imwrite_u(os.path.join(FIG_REL, f"resultado_{_sanitize(folder)}.png"), im)


def build_all_report_steps():
    """Copia todos os 9 passos de cada imagem com prefixo para o relatorio."""
    for folder in sorted(os.listdir(OUT)):
        pd = os.path.join(OUT, folder, "passos")
        if not os.path.isdir(pd):
            continue
        san = _sanitize(folder)
        for i, s in enumerate(STEPS, 1):
            im = imread_u(os.path.join(pd, s + ".png"))
            if im is not None:
                imwrite_u(os.path.join(FIG_REL, f"{san}_passo_{i}.png"), im)


def main():
    folders = [d for d in sorted(os.listdir(OUT))
               if os.path.isdir(os.path.join(OUT, d, "passos"))]
    print(f"{len(folders)} imagens")
    for f in folders:
        print("grid:", build_step_grid(f))
    build_dataset_example()
    build_report_steps()
    build_result_overlays()
    build_all_report_steps()
    print("figuras geradas em latex/figuras/ e latex/apresentacao/figuras/")


if __name__ == "__main__":
    main()
