"""
Exportacao do grafo viario (Fase 5 do plano v2).

Gera:
  - JSON estruturado (vertices + arestas com pavimento, largura, comprimento,
    confianca, flags de ponte/transicao e polyline simplificada por RDP)
  - GraphML via networkx (opcional; ignorado com aviso se networkx faltar)
"""

import json
import math

import numpy as np

from pipeline.graph_refine import SURFACE_NAMES


def rdp_simplify(points, eps=2.0):
    """Ramer-Douglas-Peucker iterativo sobre lista [(y, x), ...]."""
    if len(points) < 3:
        return list(points)
    pts = np.asarray(points, np.float32)
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a, b = pts[i0], pts[i1]
        ab = b - a
        lab = float(np.hypot(ab[0], ab[1]))
        seg = pts[i0 + 1:i1]
        if lab < 1e-6:
            d = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            d = np.abs(ab[0] * (a[1] - seg[:, 1]) - ab[1] * (a[0] - seg[:, 0])) / lab
        k = int(np.argmax(d))
        if float(d[k]) > eps:
            idx = i0 + 1 + k
            keep[idx] = True
            stack.append((i0, idx))
            stack.append((idx, i1))
    return [tuple(p) for p in np.asarray(points)[keep]]


def _node_type(G, nid):
    if G.nodes[nid].get("transition"):
        return "transition"
    d = G.degree(nid)
    if d == 1:
        return "endpoint"
    if d >= 3:
        return "intersection"
    return "waypoint"


def graph_to_dict(G, source_image=None, extra_meta=None):
    meta = {
        "source_image": source_image,
        "n_vertices": len(G.nodes),
        "n_edges": len(G.edges),
        "surface_legend": SURFACE_NAMES,
    }
    if extra_meta:
        meta.update(extra_meta)

    vertices = []
    for nid, nd in sorted(G.nodes.items()):
        vertices.append({
            "node_id": int(nid),
            "x": round(float(nd["pos"][1]), 1),
            "y": round(float(nd["pos"][0]), 1),
            "type": _node_type(G, nid),
        })

    edges = []
    for eid, e in sorted(G.edges.items()):
        poly = rdp_simplify(e["path"], eps=2.0)
        item = {
            "edge_id": int(eid),
            "source": int(e["u"]),
            "target": int(e["v"]),
            "surface": SURFACE_NAMES.get(e["surface"], "asfalto"),
            "width_px": round(float(e["width"]), 1),
            "length_px": round(float(e["length"]), 1),
            "confidence": round(float(e.get("conf", 0.0)), 3),
            "polyline": [[round(float(x), 1), round(float(y), 1)]
                         for (y, x) in poly],
        }
        if e.get("bridge"):
            item["bridge"] = True
            item["bridge_avg_cost"] = round(float(e.get("avg_cost", 0.0)), 2)
        edges.append(item)

    return {"graph_metadata": meta, "vertices": vertices, "edges": edges}


def save_graph_json(G, path, source_image=None, extra_meta=None):
    data = graph_to_dict(G, source_image, extra_meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def save_graph_graphml(G, path):
    try:
        import networkx as nx
    except ImportError:
        print("networkx indisponivel — GraphML nao gerado.")
        return False
    NG = nx.MultiGraph()
    for nid, nd in G.nodes.items():
        NG.add_node(int(nid), x=float(nd["pos"][1]), y=float(nd["pos"][0]),
                    type=_node_type(G, nid))
    for eid, e in G.edges.items():
        NG.add_edge(int(e["u"]), int(e["v"]), key=int(eid),
                    surface=SURFACE_NAMES.get(e["surface"], "asfalto"),
                    width_px=float(e["width"]),
                    length_px=float(e["length"]),
                    confidence=float(e.get("conf", 0.0)),
                    bridge=bool(e.get("bridge", False)))
    nx.write_graphml(NG, path)
    return True
