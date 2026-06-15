"""
Pos-processamento topologico unificado em grafo (Fases 2 e 3 do plano v2).

O grafo e a fonte unica de verdade apos a binarizacao: pontes, podas,
classificacao de pavimento, desenho do overlay e exportacao derivam todos
da mesma estrutura — elimina o descompasso entre mapa de probabilidade e
surface_map do pipeline legado.

Substitui (modo 'graph'): bridge_gaps.py, close_road_gaps,
prune_dangling_stubs (fase 1) e stub_smart.py.

Etapas de refine_graph():
  1. merge_close_nodes      — funde juncoes duplicadas (raio ~ largura/2)
  2. prune_stubs            — remove pontas curtas e pontas que morrem em
                              telhado/predio ("via dentro de casa")
  3. bridge_gaps_astar      — fecha gaps com caminho de menor custo sobre
                              custo = 1/(prob+eps) + contexto (agua/telhado
                              bloqueiam; vegetacao encarece)
  4. remove_small_components— descarta fragmentos isolados curtos
  5. dissolve_degree2_nodes — funde cadeias em arestas maximas entre junções

Classificacao de superficie (por ARESTA, nunca por pixel):
  classify_edges            — mediana da evidencia de terra ao longo da via
  split_surface_transitions — divide aresta longa com mudanca real de
                              pavimento (ponto de mudanca por SSE em 2 segmentos)
  icm_smooth                — suavizacao tipo Potts entre arestas continuas
                              (grau 2 e pares colineares em junções)
"""

import math
import heapq
from collections import deque

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Cores RGB de saida (terra = LARANJA, asfalto = AZUL)
COLORS_RGB = {1: (0, 100, 255), 2: (255, 140, 0)}
SURFACE_NAMES = {1: "asfalto", 2: "terra"}

# Limiar de terra por aresta. Mais alto que o operating point pixel-a-pixel
# legado (0.40): em vias finas a amostragem pega margens (grama/solo) e a
# janela de textura 7x7 transborda a via — feedback de QA mostrou asfalto
# fino virando laranja. Combinado com percentil 35 na agregacao (nucleo da
# via), o vies de margem e removido sem perder estrada de terra real.
TERRA_THR = 0.45


def _polyline_len(path):
    if len(path) < 2:
        return 0.0
    arr = np.asarray(path, np.float32)
    d = np.diff(arr, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


class RoadGraph:
    """Grafo viario: nos = junções/pontas; arestas = polylines de centerline."""

    def __init__(self, shape):
        self.shape = shape
        self.nodes = {}   # nid -> {"pos": (y, x), flags...}
        self.edges = {}   # eid -> {"u","v","path","width","length","surface","conf", flags...}
        self.adj = {}     # nid -> set(eid)
        self._nid = 0
        self._eid = 0

    # ----------------------------------------------------------- mutacao
    def add_node(self, pos, **attrs):
        nid = self._nid
        self._nid += 1
        self.nodes[nid] = {"pos": (float(pos[0]), float(pos[1])), **attrs}
        self.adj[nid] = set()
        return nid

    def add_edge(self, u, v, path, width, **attrs):
        eid = self._eid
        self._eid += 1
        path = [(int(p[0]), int(p[1])) for p in path]
        self.edges[eid] = {
            "u": u, "v": v, "path": path,
            "width": max(float(width), 2.0),
            "length": _polyline_len(path),
            "surface": 1, "conf": 0.0,
            **attrs,
        }
        self.adj[u].add(eid)
        self.adj[v].add(eid)
        return eid

    def remove_edge(self, eid, drop_orphans=True):
        e = self.edges.pop(eid, None)
        if e is None:
            return
        for nid in (e["u"], e["v"]):
            if nid in self.adj:
                self.adj[nid].discard(eid)
        if drop_orphans:
            for nid in {e["u"], e["v"]}:
                self.drop_if_orphan(nid)

    def drop_if_orphan(self, nid):
        if nid in self.adj and not self.adj[nid]:
            del self.adj[nid]
            del self.nodes[nid]

    def split_edge(self, eid, idx, **node_attrs):
        """Divide a aresta no indice idx do path. Retorna (nid_novo, (eid1, eid2))."""
        e = self.edges[eid]
        path = e["path"]
        idx = int(np.clip(idx, 1, len(path) - 2))
        extra = {k: v for k, v in e.items()
                 if k not in ("u", "v", "path", "length", "p_series")}
        nid = self.add_node(path[idx], **node_attrs)
        u, v = e["u"], e["v"]
        self.remove_edge(eid, drop_orphans=False)
        e1 = self.add_edge(u, nid, path[:idx + 1], **extra)
        e2 = self.add_edge(nid, v, path[idx:], **extra)
        return nid, (e1, e2)

    # ----------------------------------------------------------- consulta
    def degree(self, nid):
        d = 0
        for eid in self.adj.get(nid, ()):
            e = self.edges[eid]
            d += 2 if e["u"] == e["v"] else 1
        return d

    def endpoints(self):
        return [nid for nid in self.nodes if self.degree(nid) == 1]

    def components(self):
        """Lista de (set_nids, set_eids) por componente conectado."""
        seen = set()
        comps = []
        for start in self.nodes:
            if start in seen:
                continue
            nids, eids = set(), set()
            q = deque([start])
            seen.add(start)
            while q:
                nid = q.popleft()
                nids.add(nid)
                for eid in self.adj[nid]:
                    eids.add(eid)
                    e = self.edges[eid]
                    for other in (e["u"], e["v"]):
                        if other not in seen:
                            seen.add(other)
                            q.append(other)
            comps.append((nids, eids))
        return comps

    def total_length(self):
        return sum(e["length"] for e in self.edges.values())


# ---------------------------------------------------------------------------
# Construcao a partir da mascara binaria
# ---------------------------------------------------------------------------

def _skeleton(binary):
    if skeletonize is not None:
        return skeletonize(binary > 0).astype(np.uint8)
    # fallback morfologico (qualidade inferior; skimage no requirements)
    img = (binary > 0).astype(np.uint8) * 255
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros_like(img)
    tmp = img.copy()
    for _ in range(200):
        eroded = cv2.erode(tmp, elem)
        skel = cv2.bitwise_or(skel, cv2.subtract(tmp, cv2.dilate(eroded, elem)))
        tmp = eroded
        if cv2.countNonZero(tmp) == 0:
            break
    return (skel > 0).astype(np.uint8)


def graph_from_mask(binary, width_map=None):
    if width_map is None:
        width_map = cv2.distanceTransform(
            (binary > 0).astype(np.uint8), cv2.DIST_L2, 5) * 2.0
    skel = _skeleton(binary)
    return graph_from_skeleton(skel, width_map)


def graph_from_skeleton(skel, width_map):
    skel = (skel > 0).astype(np.uint8)
    h, w = skel.shape
    k8 = np.ones((3, 3), np.uint8)
    k8[1, 1] = 0
    nb = cv2.filter2D(skel, -1, k8)
    is_skel = skel > 0
    vert = (is_skel & ((nb == 1) | (nb >= 3))).astype(np.uint8)

    n_cl, v_lbl = cv2.connectedComponents(vert, connectivity=8)
    G = RoadGraph((h, w))

    cluster_pixels = {}
    ys, xs = np.nonzero(vert)
    for y, x in zip(ys.tolist(), xs.tolist()):
        cluster_pixels.setdefault(int(v_lbl[y, x]), []).append((y, x))

    node_of_cluster = {}
    for cid, pxs in cluster_pixels.items():
        cy = sum(p[0] for p in pxs) / len(pxs)
        cx = sum(p[1] for p in pxs) / len(pxs)
        node_of_cluster[cid] = G.add_node((cy, cx))

    def _edge_width(path):
        ws = [float(width_map[p]) for p in path if width_map[p] > 0]
        return float(np.median(ws)) if ws else 4.0

    used = np.zeros((h, w), bool)
    direct_pairs = set()

    for cid, pxs in cluster_pixels.items():
        for (sy, sx) in pxs:
            for dy, dx in DIRS8:
                ny, nx = sy + dy, sx + dx
                if not (0 <= ny < h and 0 <= nx < w) or not is_skel[ny, nx]:
                    continue
                if vert[ny, nx]:
                    ocid = int(v_lbl[ny, nx])
                    if ocid != cid:
                        key = (min(cid, ocid), max(cid, ocid))
                        if key not in direct_pairs:
                            direct_pairs.add(key)
                            p = [(sy, sx), (ny, nx)]
                            G.add_edge(node_of_cluster[cid], node_of_cluster[ocid],
                                       p, _edge_width(p))
                    continue
                if used[ny, nx]:
                    continue

                # caminhar pela cadeia grau-2 ate a proxima junçao/ponta
                path = [(sy, sx), (ny, nx)]
                used[ny, nx] = True
                walk_set = {(sy, sx), (ny, nx)}
                cy_, cx_ = ny, nx
                end_node = None
                while True:
                    nxt = None
                    hit = None
                    for dy2, dx2 in DIRS8:
                        ty, tx = cy_ + dy2, cx_ + dx2
                        if not (0 <= ty < h and 0 <= tx < w) or not is_skel[ty, tx]:
                            continue
                        if (ty, tx) in walk_set:
                            continue
                        if vert[ty, tx]:
                            hit = (ty, tx)
                            break
                        if nxt is None and not used[ty, tx]:
                            nxt = (ty, tx)
                    if hit is not None:
                        path.append(hit)
                        end_node = node_of_cluster[int(v_lbl[hit])]
                        break
                    if nxt is None:
                        end_node = G.add_node((float(cy_), float(cx_)))
                        break
                    used[nxt] = True
                    walk_set.add(nxt)
                    path.append(nxt)
                    cy_, cx_ = nxt

                u = node_of_cluster[cid]
                if end_node == u and len(path) < 5:
                    continue  # micro-loop de artefato de esqueletizacao
                G.add_edge(u, end_node, path, _edge_width(path))

    # Ciclos puros (ex.: rotatoria isolada): todo pixel tem grau 2 — nenhum
    # vertice foi criado e o tracer principal nunca os visita.
    rem_ys, rem_xs = np.nonzero(is_skel & (~used) & (vert == 0))
    for ry, rx in zip(rem_ys.tolist(), rem_xs.tolist()):
        if used[ry, rx]:
            continue
        start_nid = G.add_node((float(ry), float(rx)))
        used[ry, rx] = True
        path = [(ry, rx)]
        walk_set = {(ry, rx)}
        cy_, cx_ = ry, rx
        while True:
            nxt = None
            for dy2, dx2 in DIRS8:
                ty, tx = cy_ + dy2, cx_ + dx2
                if (0 <= ty < h and 0 <= tx < w and is_skel[ty, tx]
                        and not vert[ty, tx] and not used[ty, tx]
                        and (ty, tx) not in walk_set):
                    nxt = (ty, tx)
                    break
            if nxt is None:
                break
            used[nxt] = True
            walk_set.add(nxt)
            path.append(nxt)
            cy_, cx_ = nxt
        if len(path) < 4:
            G.drop_if_orphan(start_nid)
            continue
        # fecha o anel se o fim e adjacente ao inicio
        ey, ex = path[-1]
        if max(abs(ey - ry), abs(ex - rx)) <= 1:
            path.append((ry, rx))
        G.add_edge(start_nid, start_nid, path, _edge_width(path))
    return G


# ---------------------------------------------------------------------------
# Refinamento topologico
# ---------------------------------------------------------------------------

def merge_close_nodes(G, radius):
    cell = max(radius, 1.0)
    buckets = {}
    for nid, nd in G.nodes.items():
        key = (int(nd["pos"][0] // cell), int(nd["pos"][1] // cell))
        buckets.setdefault(key, []).append(nid)

    parent = {nid: nid for nid in G.nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    r2 = radius * radius
    for key, ids in buckets.items():
        cand = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cand += buckets.get((key[0] + dy, key[1] + dx), [])
        for a in ids:
            pa = G.nodes[a]["pos"]
            for b in cand:
                if b <= a:
                    continue
                pb = G.nodes[b]["pos"]
                if (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 <= r2:
                    union(a, b)

    groups = {}
    for nid in list(G.nodes):
        groups.setdefault(find(nid), []).append(nid)

    for root, members in groups.items():
        if len(members) == 1:
            continue
        py = sum(G.nodes[m]["pos"][0] for m in members) / len(members)
        px = sum(G.nodes[m]["pos"][1] for m in members) / len(members)
        G.nodes[root]["pos"] = (py, px)
        for m in members:
            if m == root:
                continue
            if G.nodes[m].get("keep"):
                G.nodes[root]["keep"] = True
            for eid in list(G.adj[m]):
                e = G.edges[eid]
                if e["u"] == m:
                    e["u"] = root
                if e["v"] == m:
                    e["v"] = root
                G.adj[root].add(eid)
            del G.adj[m]
            del G.nodes[m]

    # remove self-loops degenerados criados pela fusao
    for eid in list(G.edges):
        e = G.edges[eid]
        if e["u"] == e["v"] and e["length"] < 3 * radius:
            G.remove_edge(eid)


def _frac_in_mask(mask, pos, r=3):
    h, w = mask.shape
    y, x = int(round(pos[0])), int(round(pos[1]))
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    win = mask[y0:y1, x0:x1]
    return float(win.mean()) if win.size else 0.0


def prune_stubs(G, typical_width, roof_mask=None, border_margin=8, max_rounds=8):
    """Remove pontas curtas e pontas que terminam dentro de construcao."""
    h, w = G.shape
    removed = 0
    for _ in range(max_rounds):
        changed = False
        for nid in G.endpoints():
            if nid not in G.nodes:
                continue
            eid = next(iter(G.adj[nid]))
            e = G.edges[eid]
            y, x = G.nodes[nid]["pos"]
            if (y < border_margin or x < border_margin or
                    y >= h - border_margin or x >= w - border_margin):
                continue  # via truncada pela borda: termino valido
            if e.get("bridge"):
                continue
            # Pontas curtas que terminam no meio do quarteirao sao ruido
            # (textura de telhado, beco que nao e rua). Uma rua de verdade nao
            # morre num toco de 15-50px longe da borda. Discriminador = comprimento
            # (seguro entre dominios: estrada de terra real e LONGA, nao um toco).
            min_short = min(max(20.0, 3.0 * e["width"]), 55.0)
            if e["length"] < min_short:
                G.remove_edge(eid)
                removed += 1
                changed = True
                continue
            if roof_mask is not None and _frac_in_mask(roof_mask, (y, x), 3) > 0.3 \
                    and e["length"] < 4 * min_short:
                # rua nao termina em cima de telhado
                G.remove_edge(eid)
                removed += 1
                changed = True
        if not changed:
            break
    return removed


def _edge_dir_at_node(G, eid, nid, k=8):
    """Tangente unitaria (dy,dx) da aresta no no, apontando para FORA da via."""
    e = G.edges[eid]
    path = e["path"]
    if len(path) < 2:
        return None
    if e["u"] == nid:
        a = np.float32(path[0])
        b = np.float32(path[min(k, len(path) - 1)])
    else:
        a = np.float32(path[-1])
        b = np.float32(path[max(0, len(path) - 1 - k)])
    v = a - b
    n = float(np.hypot(v[0], v[1]))
    if n < 0.5:
        return None
    return (float(v[0] / n), float(v[1] / n))


def _centerline_index(G):
    lab = np.full(G.shape, -1, np.int32)
    pxmap = {}
    for eid, e in G.edges.items():
        for i, p in enumerate(e["path"]):
            lab[p] = eid
            pxmap[p] = (eid, i)
    return lab, pxmap


def _resolve_hit_px(lab, cy, cx, own_eid):
    """Pixel rotulado em (cy,cx) ou na vizinhanca-8 (caminhos 8-conexos podem
    se cruzar na diagonal sem compartilhar pixel)."""
    h, w = lab.shape
    best = None
    for dy, dx in [(0, 0)] + DIRS8:
        ty, tx = cy + dy, cx + dx
        if 0 <= ty < h and 0 <= tx < w and lab[ty, tx] >= 0:
            if int(lab[ty, tx]) != own_eid:
                return (ty, tx)
            if best is None:
                best = (ty, tx)
    return best


def _dijkstra_to_road(cost, blocked, lab, hit_zone, start, dirv, own_eid, R,
                      cone_deg=70.0, max_pop=20000):
    """Caminho de menor custo do endpoint ate qualquer centerline do grafo."""
    h, w = cost.shape
    sy, sx = start
    dy0, dx0 = dirv
    y0w, y1w = max(0, sy - R), min(h, sy + R + 1)
    x0w, x1w = max(0, sx - R), min(w, sx + R + 1)
    cos_lim = math.cos(math.radians(cone_deg))

    dist = {(sy, sx): 0.0}
    steps_of = {(sy, sx): 0}
    came = {}
    pq = [(0.0, sy, sx)]
    pops = 0

    while pq:
        g, cy, cx = heapq.heappop(pq)
        pops += 1
        if pops > max_pop:
            return None
        if dist.get((cy, cx), 1e18) < g - 1e-9:
            continue
        st = steps_of[(cy, cx)]
        if st > 8 and hit_zone[cy, cx]:
            hit = _resolve_hit_px(lab, cy, cx, own_eid)
            if hit is not None:
                l = int(lab[hit])
                if l != own_eid or st > 25:
                    path = []
                    cur = (cy, cx)
                    while cur != (sy, sx):
                        path.append(cur)
                        cur = came[cur]
                    path.reverse()
                    return path, g, hit
        if st >= R:
            continue
        for dy, dx in DIRS8:
            ny, nx = cy + dy, cx + dx
            if not (y0w <= ny < y1w and x0w <= nx < x1w):
                continue
            if blocked[ny, nx]:
                continue
            vy, vx = ny - sy, nx - sx
            nrm = math.hypot(vy, vx)
            if nrm > 2.0 and (vy * dy0 + vx * dx0) / nrm < cos_lim:
                continue  # fora do cone direcional
            ng = g + float(cost[ny, nx]) * (1.4142 if (dy and dx) else 1.0)
            if ng < dist.get((ny, nx), 1e18):
                dist[(ny, nx)] = ng
                steps_of[(ny, nx)] = st + 1
                came[(ny, nx)] = (cy, cx)
                heapq.heappush(pq, (ng, ny, nx))
    return None


def bridge_gaps_astar(G, prob, masks, typical_width, rounds=2,
                      avg_cost_max=12.0, straight_avg_max=30.0,
                      short_avg_max=45.0, tortuosity_max=1.8,
                      tort_straight=1.25, border_margin=8):
    """
    Fecha gaps por caminho de menor custo sobre c = 1/(prob+eps) + contexto.

    Aceitacao em tres niveis (calibrada para reproduzir o comportamento do
    bridging legado, que aceitava linha reta com sinal medio >= 0.012):
      - salto curto  (steps <= 2x largura tipica): avg <= 45 — equivale ao
        fechamento morfologico do pipeline legado;
      - ponte RETA   (tortuosidade <= 1.25): avg <= 30 — sinal residual
        fraco basta quando a continuacao e perfeitamente alinhada;
      - ponte curva  (tortuosidade <= 1.8): avg <= 12 — curva so com sinal
        forte do modelo.
    """
    h, w = G.shape
    cost = 1.0 / (np.clip(prob, 0.0, 1.0).astype(np.float32) + 0.02)
    blocked = np.zeros(prob.shape, bool)
    if masks is not None:
        cost = cost + 4.0 * (masks["vegetation"] > 0).astype(np.float32)
        cost = cost + 1.5 * (masks["soil"] > 0).astype(np.float32)
        blocked |= masks["water"] > 0
        blocked |= masks["roof"] > 0

    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    n_bridges = 0
    for _round in range(rounds):
        lab, pxmap = _centerline_index(G)
        hit_zone = cv2.dilate((lab >= 0).astype(np.uint8), k3) > 0
        new_any = False
        for nid in list(G.endpoints()):
            if nid not in G.nodes:
                continue
            eid0 = next(iter(G.adj[nid]))
            e0 = G.edges[eid0]
            d = _edge_dir_at_node(G, eid0, nid)
            if d is None:
                continue
            py, px_ = G.nodes[nid]["pos"]
            sy, sx = int(round(py)), int(round(px_))
            if (sy < border_margin or sx < border_margin or
                    sy >= h - border_margin or sx >= w - border_margin):
                continue
            R = int(np.clip(12.0 * max(e0["width"], typical_width), 80, 300))
            max_pop = min(80000, (2 * R + 1) * (2 * R + 1))
            res = _dijkstra_to_road(cost, blocked, lab, hit_zone, (sy, sx), d,
                                    eid0, R, max_pop=max_pop)
            if res is None:
                continue
            path, g_total, hit = res
            steps = len(path)
            avg = g_total / max(steps, 1)
            eucl = math.hypot(path[-1][0] - sy, path[-1][1] - sx)
            tort = steps / max(eucl, 1.0)
            s_short = max(12.0, 2.0 * typical_width)
            ok = (steps <= s_short and avg <= short_avg_max) or \
                 (tort <= tort_straight and avg <= straight_avg_max) or \
                 (tort <= tortuosity_max and avg <= avg_cost_max)
            if not ok:
                continue

            t_eid, t_idx = pxmap.get(hit, (int(lab[hit]), None))
            te = G.edges.get(t_eid)
            if te is None or t_idx is None:
                continue
            if t_idx <= 2:
                t_node = te["u"]
            elif t_idx >= len(te["path"]) - 3:
                t_node = te["v"]
            else:
                t_node, _ = G.split_edge(t_eid, t_idx)
            if t_node == nid:
                continue
            G.add_edge(nid, t_node, [(sy, sx)] + path, e0["width"],
                       bridge=True, avg_cost=float(avg))
            n_bridges += 1
            new_any = True
            lab, pxmap = _centerline_index(G)  # indices mudaram (split/nova aresta)
            hit_zone = cv2.dilate((lab >= 0).astype(np.uint8), k3) > 0
        if not new_any:
            break
    return {"bridges": n_bridges}


def remove_small_components(G, min_total_len, border_margin=8):
    h, w = G.shape
    removed = 0
    for nids, eids in G.components():
        total = sum(G.edges[eid]["length"] for eid in eids)
        if total >= min_total_len:
            continue
        touches_border = any(
            G.nodes[nid]["pos"][0] < border_margin or
            G.nodes[nid]["pos"][1] < border_margin or
            G.nodes[nid]["pos"][0] >= h - border_margin or
            G.nodes[nid]["pos"][1] >= w - border_margin
            for nid in nids)
        if touches_border:
            continue  # fragmento na borda: via que continua fora da imagem
        for eid in eids:
            G.remove_edge(eid)
            removed += 1
    return removed


def dissolve_degree2_nodes(G):
    """Funde cadeias em arestas maximas entre junções (nos 'keep' preservados)."""
    changed = True
    while changed:
        changed = False
        for nid in list(G.nodes):
            if nid not in G.nodes or G.nodes[nid].get("keep"):
                continue
            if G.degree(nid) != 2 or len(G.adj[nid]) != 2:
                continue
            e1id, e2id = tuple(G.adj[nid])
            e1, e2 = G.edges[e1id], G.edges[e2id]
            p1 = e1["path"] if e1["v"] == nid else e1["path"][::-1]  # o1 -> nid
            o1 = e1["u"] if e1["v"] == nid else e1["v"]
            p2 = e2["path"] if e2["u"] == nid else e2["path"][::-1]  # nid -> o2
            o2 = e2["v"] if e2["u"] == nid else e2["u"]
            if o1 == nid or o2 == nid:
                continue
            new_path = p1 + p2[1:]
            L = e1["length"] + e2["length"]
            width = (e1["width"] * e1["length"] + e2["width"] * e2["length"]) / max(L, 1e-6)
            attrs = {}
            if e1.get("bridge") or e2.get("bridge"):
                attrs["bridge"] = True
            G.remove_edge(e1id, drop_orphans=False)
            G.remove_edge(e2id, drop_orphans=False)
            G.drop_if_orphan(nid)
            G.add_edge(o1, o2, new_path, width, **attrs)
            changed = True


def _still_connected(G, u, v, exclude):
    """u alcanca v sem usar a aresta `exclude`? (BFS sobre o grafo)."""
    seen = {u}
    stack = [u]
    while stack:
        cur = stack.pop()
        if cur == v:
            return True
        for eid in G.adj.get(cur, ()):
            if eid == exclude:
                continue
            e = G.edges[eid]
            other = e["v"] if e["u"] == cur else e["u"]
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return False


def remove_short_loop_edges(G, max_len):
    """
    Remove arestas CURTAS e REDUNDANTES — as que fecham 'quarteiroes falsos'
    dentro de ruas (conectores internos sobre telhados/textura).

    Seguranca: so remove se os dois extremos continuam ligados por OUTRO
    caminho (a aresta faz parte de um ciclo). Logo nunca desconecta a malha:
    ruas de verdade sao 'pontes' no sentido de grafo (remove-las desconectaria)
    e por isso sao mantidas. Exige extremos com grau >= 3 (conector interno,
    nao ponta solta — pontas sao tratadas por prune_stubs).
    """
    removed = 0
    for eid in list(G.edges):
        e = G.edges.get(eid)
        if e is None or e["u"] == e["v"] or e.get("bridge"):
            continue
        if e["length"] >= max_len:
            continue
        u, v = e["u"], e["v"]
        if G.degree(u) < 3 or G.degree(v) < 3:
            continue
        if _still_connected(G, u, v, exclude=eid):
            G.remove_edge(eid)
            removed += 1
    return removed


def _roof_min_sides(roof_mask, min_block_area=400):
    """Rotula os quarteiroes (componentes do telhado) e mede o MENOR LADO
    (retangulo minimo) de cada um."""
    rm = (roof_mask > 0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(rm, connectivity=8)
    min_side = {}
    for bid in range(1, n):
        cw = int(stats[bid, cv2.CC_STAT_WIDTH]); ch = int(stats[bid, cv2.CC_STAT_HEIGHT])
        if cw * ch < min_block_area:
            continue
        x0 = int(stats[bid, cv2.CC_STAT_LEFT]); y0 = int(stats[bid, cv2.CC_STAT_TOP])
        comp = (lbl[y0:y0 + ch, x0:x0 + cw] == bid).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        (rw, rh) = cv2.minAreaRect(max(cnts, key=cv2.contourArea))[1]
        min_side[bid] = min(rw, rh)
    return lbl, min_side


def remove_edges_inside_blocks(G, roof_mask, typical_width, min_span_frac=0.7,
                               prob=None, keep_prob_thr=0.30, min_roof_frac=0.28):
    """
    Remove TRACOS INTERNOS de quarteirao (regra do usuario):

      - a estrada tem UMA PONTA SOLTA (no de grau 1 — uma ponta sai de outra
        estrada e a outra nao liga em nada, como rua sem saida);  E
      - essa ponta livre esta DENTRO de um quarteirao — cercada por telhado
        (um disco ao redor dela e majoritariamente bloco);  E
      - o ALCANCE da estrada (ponta-a-ponta) e MENOR que `min_span_frac` (70%)
        do MENOR LADO do retangulo do quarteirao.

    Entao a estrada nao atravessa o bloco — e um traco interno (sombra, vao,
    beco sem saida sobre telhado) — e e removida (nao vira azul). Estradas que
    ligam dois cruzamentos (sem ponta solta) ou que cruzam >= 70% do bloco sao
    mantidas. Iterativo: remover um traco pode expor outra ponta solta.

    Deteccao por DISCO (nao por preenchimento de buracos): a ponta do toco e um
    vao carvado do telhado que se conecta a rua, entao nunca e um buraco
    fechado; mas a regiao AO REDOR dela e o bloco. Por isso medimos a fracao de
    telhado num disco centrado na ponta livre.
    """
    if roof_mask is None or int(roof_mask.sum()) == 0:
        return 0
    lbl, min_side = _roof_min_sides(roof_mask)
    if not min_side:
        return 0
    h, w = roof_mask.shape
    R = max(8, int(round(2.0 * typical_width)))  # raio do disco ao redor da ponta
    removed = 0
    _dbg = bool(__import__("os").environ.get("DBG_BLK"))
    for _ in range(8):
        changed = False
        for nid in list(G.endpoints()):          # grau 1 = ponta solta
            if nid not in G.nodes:
                continue
            eid = next(iter(G.adj[nid]))
            e = G.edges.get(eid)
            if e is None or e.get("bridge"):
                continue
            y, x = G.nodes[nid]["pos"]
            iy, ix = int(round(y)), int(round(x))
            y0, y1 = max(0, iy - R), min(h, iy + R + 1)
            x0, x1 = max(0, ix - R), min(w, ix + R + 1)
            win = lbl[y0:y1, x0:x1]
            roofpix = win[win > 0]
            roof_frac = roofpix.size / max(win.size, 1)
            (ay, ax), (by, bx) = e["path"][0], e["path"][-1]
            span = math.hypot(by - ay, bx - ax)  # alcance ponta-a-ponta
            ms = 0.0
            if roofpix.size:
                vals, cnts = np.unique(roofpix, return_counts=True)
                bid = int(vals[np.argmax(cnts)])     # bloco dominante ao redor
                ms = float(min_side.get(bid, 0) or 0)
            medp = -1.0
            if prob is not None:
                pp = [float(prob[int(round(py)), int(round(px))])
                      for (py, px) in e["path"]
                      if 0 <= int(round(py)) < prob.shape[0]
                      and 0 <= int(round(px)) < prob.shape[1]]
                medp = float(np.median(pp)) if pp else -1.0
            if _dbg:
                print(f"  EP {nid} pos=({iy},{ix}) roof_frac={roof_frac:.2f} "
                      f"span={span:.0f} ms={ms:.0f} thr_span={min_span_frac*ms:.0f}"
                      f" medp={medp:.2f} len={len(e['path'])}")
            # ponta dentro/junto de um bloco? (parte do disco e telhado). O
            # limiar pode ser BAIXO (0.28) porque o discriminador forte e a
            # PROBABILIDADE abaixo: rua real e forte, toco interno e fraco. Isso
            # mantem protegidas as "ruas reais que encostam num predio" (fortes),
            # que antes sumiam quando o disco era a unica regra.
            if roof_frac < min_roof_frac:
                continue
            if not ms or ms <= 0:
                continue
            if span >= min_span_frac * ms:
                continue
            # portao de probabilidade (PRINCIPAL): uma rua REAL e forte (>= seed);
            # um toco interno de quarteirao so existe pela hysteresis baixa
            # (fraco). Nunca remove via forte, mesmo caindo dentro de um bloco.
            if prob is not None and medp >= keep_prob_thr:
                continue
            G.remove_edge(eid)
            removed += 1
            changed = True
        if not changed:
            break
    return removed


def refine_graph(G, prob, masks, typical_width, do_bridge=True,
                 block_filter=True, toco_frac=0.80):
    stats = {"edges_in": len(G.edges)}
    merge_close_nodes(G, radius=max(3.0, typical_width * 0.5))
    stats["stubs_removed"] = prune_stubs(
        G, typical_width, masks["roof"] if masks else None)
    if do_bridge and len(G.edges) > 0:
        stats.update(bridge_gaps_astar(G, prob, masks, typical_width))
        stats["stubs_removed_pos_ponte"] = prune_stubs(
            G, typical_width, masks["roof"] if masks else None, max_rounds=3)
    # quebra 'quarteiroes falsos' (loops curtos internos) e re-poda os tocos
    # que sobrarem
    stats["loops_removed"] = remove_short_loop_edges(
        G, max_len=max(28.0, 4.5 * typical_width))
    stats["stubs_removed_pos_loop"] = prune_stubs(
        G, typical_width, masks["roof"] if masks else None, max_rounds=4)
    # remove TRACOS INTERNOS de quarteirao: ponta solta dentro do bloco e
    # alcance < 70% do menor lado do poligono (regra do usuario). Roda apos a
    # poda e a dissolucao para medir o alcance real de cada traco.
    dissolve_degree2_nodes(G)
    if block_filter and masks is not None:
        stats["inside_block_removed"] = remove_edges_inside_blocks(
            G, masks.get("roof_solid", masks.get("roof")), typical_width,
            min_span_frac=toco_frac, prob=prob)
    stats["components_removed"] = remove_small_components(
        G, min_total_len=max(40.0, 4.0 * typical_width))
    dissolve_degree2_nodes(G)
    stats["edges_out"] = len(G.edges)
    stats["nodes_out"] = len(G.nodes)
    return stats


# ---------------------------------------------------------------------------
# Classificacao de superficie por aresta (Fase 3)
# ---------------------------------------------------------------------------

def _classify_from_series(e, series, thr=TERRA_THR):
    # Mediana da evidencia de terra ao longo do NUCLEO da via (o raio apertado
    # em classify_edges ja evita amostrar margens quentes). O limiar `thr` e
    # adaptado por dominio (urbano alto ~0.45 / rural baixo ~0.25) por quem
    # chama, resolvendo a sobreposicao terra-clara x asfalto-cinza.
    agg = float(np.median(series)) if len(series) else 0.0
    e["surface"] = 2 if agg >= thr else 1
    e["obs_surface"] = e["surface"]
    e["conf"] = float(np.clip(abs(agg - thr) / 0.25, 0.05, 1.0))
    e["p_terra_med"] = agg


def flip_isolated_terra(G, passes=2):
    """
    Uma aresta de TERRA cercada SO por asfalto (todas as arestas vizinhas, nos
    dois extremos, sao asfalto) e quase sempre erro: rua de terra de verdade e
    um trecho conectado, nao uma ilha de terra no meio do asfalto. Vira asfalto.
    Resolve os 'tocos soltos de terra no meio de via asfaltada'.
    """
    flips = 0
    for _ in range(passes):
        changed = 0
        for eid, e in list(G.edges.items()):
            if e["surface"] != 2:
                continue
            adj = set()
            for nid in (e["u"], e["v"]):
                for x in G.adj.get(nid, ()):
                    if x != eid:
                        adj.add(x)
            if adj and all(G.edges[x]["surface"] == 1 for x in adj):
                e["surface"] = 1
                flips += 1
                changed += 1
        if changed == 0:
            break
    return flips


def classify_edges(G, p_terra, thr=TERRA_THR, sample_step=3):
    """Amostra a evidencia de terra ao longo da centerline de cada aresta."""
    h, w = p_terra.shape
    for e in G.edges.values():
        # raio apertado: nao deixar o disco transbordar para fora da via fina
        r = int(np.clip(round(e["width"] / 2.0) - 1, 1, 4))
        samples = []
        for i in range(0, len(e["path"]), sample_step):
            y, x = e["path"][i]
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            samples.append(float(np.mean(p_terra[y0:y1, x0:x1])))
        e["p_series"] = samples
        _classify_from_series(e, samples, thr)


def split_surface_transitions(G, thr=TERRA_THR, min_seg=10, min_delta=0.22,
                              sample_step=3):
    """
    Mudanca REAL de pavimento no meio de aresta longa: ajuste em 2 segmentos
    constantes por SSE; se as medias divergirem e trocarem de classe, divide a
    aresta com um no {"transition": True} (preserva fidelidade sem zebra).
    """
    n_splits = 0
    for eid in list(G.edges):
        e = G.edges.get(eid)
        if e is None:
            continue
        s = e.get("p_series") or []
        n = len(s)
        if n < 2 * min_seg or e["length"] < 8 * e["width"]:
            continue
        arr = np.asarray(s, np.float32)
        csum = np.cumsum(arr)
        csq = np.cumsum(arr * arr)
        best_k, best_sse = -1, float("inf")
        for k in range(min_seg, n - min_seg):
            s1, q1 = csum[k - 1], csq[k - 1]
            s2, q2 = csum[-1] - s1, csq[-1] - q1
            sse = (q1 - s1 * s1 / k) + (q2 - s2 * s2 / (n - k))
            if sse < best_sse:
                best_sse, best_k = sse, k
        if best_k < 0:
            continue
        m1 = csum[best_k - 1] / best_k
        m2 = (csum[-1] - csum[best_k - 1]) / (n - best_k)
        if abs(m1 - m2) < min_delta or (m1 >= thr) == (m2 >= thr):
            continue
        path_idx = min(best_k * sample_step, len(e["path"]) - 2)
        _, (a, b) = G.split_edge(eid, path_idx, transition=True, keep=True)
        for half_eid, seg in ((a, arr[:best_k]), (b, arr[best_k:])):
            he = G.edges[half_eid]
            he["p_series"] = seg.tolist()
            _classify_from_series(he, seg, thr)
        n_splits += 1
    return n_splits


def icm_smooth(G, beta=0.5, passes=4, collinear_deg=145.0):
    """
    Suavizacao tipo Potts: arestas que continuam a MESMA via fisica
    (no de grau 2, ou par colinear em junçao) tendem a mesma classe.
    Transicoes reais (nos com flag transition) nao recebem suavizacao.
    """
    pairs = {}

    def _add_pair(a, b, wgt):
        pairs.setdefault(a, []).append((b, wgt))
        pairs.setdefault(b, []).append((a, wgt))

    cos_lim = -math.cos(math.radians(180.0 - collinear_deg))
    for nid in G.nodes:
        if G.nodes[nid].get("transition"):
            continue
        eids = list(G.adj[nid])
        if len(eids) == 2 and G.degree(nid) == 2:
            _add_pair(eids[0], eids[1], beta)
        elif len(eids) >= 3:
            dirs = []
            for eid in eids:
                d = _edge_dir_at_node(G, eid, nid)
                if d is not None:
                    dirs.append((eid, d))
            cands = []
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    di, dj = dirs[i][1], dirs[j][1]
                    cosang = di[0] * dj[0] + di[1] * dj[1]
                    # ambos apontam para fora: continuacao reta => cos ~ -1
                    if cosang <= cos_lim:
                        cands.append((cosang, dirs[i][0], dirs[j][0]))
            cands.sort()
            matched = set()
            for _, a, b in cands:
                if a in matched or b in matched:
                    continue
                matched.add(a)
                matched.add(b)
                _add_pair(a, b, beta * 0.8)

    flips = 0
    for _ in range(passes):
        changed = False
        for eid, e in G.edges.items():
            best_lbl, best_cost = e["surface"], None
            for lbl in (1, 2):
                c = e["conf"] * (0 if lbl == e.get("obs_surface", e["surface"]) else 1)
                for oeid, wgt in pairs.get(eid, ()):
                    oe = G.edges.get(oeid)
                    if oe is not None and lbl != oe["surface"]:
                        c += wgt
                if best_cost is None or c < best_cost - 1e-9:
                    best_cost, best_lbl = c, lbl
            if best_lbl != e["surface"]:
                e["surface"] = best_lbl
                changed = True
                flips += 1
        if not changed:
            break
    return flips


# ---------------------------------------------------------------------------
# Rasterizacao e desenho
# ---------------------------------------------------------------------------

def rasterize_surface(G, shape):
    """Mascara uint8 {0=fundo, 1=asfalto, 2=terra} rasterizada do grafo."""
    surf = np.zeros(shape, np.uint8)
    for cls in (1, 2):  # terra por cima nas sobreposicoes de junçao
        for e in G.edges.values():
            if e["surface"] != cls:
                continue
            r = int(np.clip(round(e["width"] / 2.0), 2, 16))
            pts = np.array([(p[1], p[0]) for p in e["path"]],
                           np.int32).reshape(-1, 1, 2)
            cv2.polylines(surf, [pts], False, int(cls),
                          thickness=2 * r + 1, lineType=cv2.LINE_8)
    # juntas suaves: disco no no com a classe/raio da aresta mais larga
    for nid, nd in G.nodes.items():
        best = None
        for eid in G.adj.get(nid, ()):
            e = G.edges[eid]
            if best is None or e["width"] > best["width"]:
                best = e
        if best is None:
            continue
        r = int(np.clip(round(best["width"] / 2.0), 2, 16))
        cv2.circle(surf, (int(round(nd["pos"][1])), int(round(nd["pos"][0]))),
                   r, int(best["surface"]), -1)
    return surf


def draw_graph_overlay(img_rgb, G, alpha_strength=0.72, blur_sigma=2.0):
    surf = rasterize_surface(G, img_rgb.shape[:2])
    canvas = np.zeros(img_rgb.shape, np.float32)
    for cls, color in COLORS_RGB.items():
        m = surf == cls
        for c in range(3):
            canvas[:, :, c][m] = color[c]
    alpha = (surf > 0).astype(np.float32)
    canvas_blur = cv2.GaussianBlur(canvas, (0, 0), sigmaX=blur_sigma)
    alpha_blur = np.clip(cv2.GaussianBlur(alpha, (0, 0), sigmaX=blur_sigma), 0, 1)
    a3 = alpha_blur[:, :, None] * alpha_strength
    out = img_rgb.astype(np.float32) * (1.0 - a3) + canvas_blur * a3
    return out.clip(0, 255).astype(np.uint8)


def graph_stats(G):
    comps = G.components()
    return {
        "nodes": len(G.nodes),
        "edges": len(G.edges),
        "endpoints": len(G.endpoints()),
        "components": len(comps),
        "total_length_px": round(G.total_length(), 1),
        "bridges": sum(1 for e in G.edges.values() if e.get("bridge")),
        "transitions": sum(1 for n in G.nodes.values() if n.get("transition")),
        "edges_terra": sum(1 for e in G.edges.values() if e["surface"] == 2),
        "edges_asfalto": sum(1 for e in G.edges.values() if e["surface"] == 1),
    }
