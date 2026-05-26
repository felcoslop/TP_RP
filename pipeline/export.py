import json
import cv2
import numpy as np

def export_json(nodes, edges, output_path="outputs/graph_output.json", img_shape=None):
    graph_data = {
        "graph_metadata": {
            "source_image": "image.png",
            "resolution_m_per_pixel": 1.0
        },
        "vertices": [],
        "edges": []
    }
    
    border_margin = 8
    
    for i, node in enumerate(nodes):
        if hasattr(node, 'tolist'):
            x, y = node.tolist()
        else:
            x, y = float(node[0]), float(node[1])
            
        is_truncated = False
        if img_shape is not None:
            h, w = img_shape[:2]
            if y < border_margin or y >= h - border_margin or x < border_margin or x >= w - border_margin:
                is_truncated = True
                
        vertex_data = {
            "node_id": i,
            "x": round(x, 1),
            "y": round(y, 1),
            "type": "intersection",
            "confidence": 0.99
        }
        
        if is_truncated:
            vertex_data["endpoint_kind"] = "truncated_at_image_edge"
            
        graph_data["vertices"].append(vertex_data)
        
    for u, v, material, prob in edges:
        if hasattr(nodes[u], 'numpy'):
            p1 = nodes[u].numpy()
            p2 = nodes[v].numpy()
        else:
            p1 = np.array([nodes[u][0], nodes[u][1]])
            p2 = np.array([nodes[v][0], nodes[v][1]])
        dist = np.linalg.norm(p1 - p2)
        graph_data["edges"].append({
            "source": u,
            "target": v,
            "surface": material,
            "probability": round(prob, 4),
            "length_px": round(float(dist), 1)
        })
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

def export_overlay(img, nodes, edges, output_path="outputs/overlay_output.png"):
    overlay = img.copy()
    COLOR_MAP = {
        "asfalto": (255, 100, 0),
        "terra": (0, 140, 255),
        "cascalho": (150, 150, 150),
        "concreto": (200, 200, 200),
        "estrada": (0, 200, 255),
        "fundo": (0, 0, 0)
    }

    for (u_idx, v_idx, material, prob) in edges:
        if hasattr(nodes[u_idx], 'int'):
            u = tuple(nodes[u_idx].int().tolist())
            v = tuple(nodes[v_idx].int().tolist())
        else:
            u = (int(nodes[u_idx][0]), int(nodes[u_idx][1]))
            v = (int(nodes[v_idx][0]), int(nodes[v_idx][1]))
        color = COLOR_MAP.get(material, (0, 255, 0))
        cv2.line(overlay, u, v, color, thickness=2)

    for node in nodes:
        if hasattr(node, 'int'):
            pt = tuple(node.int().tolist())
        else:
            pt = (int(node[0]), int(node[1]))
        cv2.circle(overlay, pt, radius=3, color=(0, 0, 255), thickness=-1)

    result = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
