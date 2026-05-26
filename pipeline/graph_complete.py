import numpy as np
import cv2
import math
import heapq
from pipeline.bridge_gaps import get_endpoints_and_directions

def extract_color_features(img_hsv, y, x, radius=3):
    h, w = img_hsv.shape[:2]
    y_min, y_max = max(0, int(y - radius)), min(h, int(y + radius + 1))
    x_min, x_max = max(0, int(x - radius)), min(w, int(x + radius + 1))
    patch = img_hsv[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return np.array([0,0,0])
    return np.mean(patch, axis=(0,1))

def color_distance(c1, c2):
    # c1, c2 are HSV (H: 0-180, S: 0-255, V: 0-255)
    dh = min(abs(c1[0] - c2[0]), 180 - abs(c1[0] - c2[0])) / 180.0
    ds = abs(c1[1] - c2[1]) / 255.0
    dv = abs(c1[2] - c2[2]) / 255.0
    # Peso maior para o Hue se a saturacao for relevante
    weight_h = 2.0 if c1[1] > 30 else 0.5
    return math.sqrt((weight_h * dh)**2 + ds**2 + dv**2)

def ray_march_endpoint(img_hsv, surface_map, start_y, start_x, dy, dx, base_color, is_terra):
    h, w = surface_map.shape
    max_steps = 250
    max_unsupported = 50 if is_terra else 30
    color_thresh = 0.25 if is_terra else 0.15 # Tolerancia HSV
    
    path = []
    cy, cx = float(start_y), float(start_x)
    cdy, cdx = dy, dx
    unsupported = 0
    hit_via = False
    
    for step in range(1, max_steps):
        cy += cdy
        cx += cdx
        iy, ix = int(round(cy)), int(round(cx))
        
        # Saiu da imagem
        if not (0 <= iy < h and 0 <= ix < w):
            break
            
        # Encontrou outra rua ja mapeada (nao no inicio)
        if step > 5 and surface_map[iy, ix] > 0:
            hit_via = True
            path.append((iy, ix))
            break
            
        # Coleta cor atual
        curr_color = extract_color_features(img_hsv, iy, ix, radius=2)
        dist = color_distance(base_color, curr_color)
        
        if dist > color_thresh:
            unsupported += 1
            # Curvar suavemente para tentar achar a estrada (wiggling)
            cdy = cdy * 0.8 + 0.2 * (np.random.random() - 0.5)
            cdx = cdx * 0.8 + 0.2 * (np.random.random() - 0.5)
            norm = math.hypot(cdy, cdx)
            if norm > 0:
                cdy, cdx = cdy/norm, cdx/norm
        else:
            unsupported = max(0, unsupported - 1)
            # Atualiza levemente a cor base para se adaptar a iluminacao
            base_color = base_color * 0.9 + curr_color * 0.1
            
        if unsupported > max_unsupported:
            break
            
        path.append((iy, ix))
        
    return path, hit_via

def complete_roads(surface_map, width_map, skel, nodes, edges, vertices, img_rgb, road_prob, tw):
    print(">>> Iniciando Graph Complete Pos-Processamento...")
    h, w = surface_map.shape
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Pegar endpoints usando funcao existente do bridge_gaps
    skel_bin = (skel > 0).astype(np.uint8)
    endpoints = get_endpoints_and_directions(skel_bin, look_back=15)
    
    new_surface = surface_map.copy()
    extensions_made = 0
    
    for ep in endpoints:
        sy, sx = ep['y'], ep['x']
        dy, dx = ep['dy'], ep['dx']
        
        # Verifica se esta no surface_map (pode ser asfalto ou terra)
        surface_type = surface_map[sy, sx]
        if surface_type == 0:
            # Tentar achar proximo
            for offset in [(0,1),(1,0),(-1,0),(0,-1),(1,1)]:
                if surface_map[sy+offset[0], sx+offset[1]] > 0:
                    surface_type = surface_map[sy+offset[0], sx+offset[1]]
                    break
        
        is_terra = (surface_type == 2)
        base_color = extract_color_features(img_hsv, sy, sx, radius=3)
        
        # Para terra, a cor Hue deve ser 'quente' (vermelho/laranja/amarelo) - H no OpenCV vai ate 180
        # H=0 a 30, ou H=150 a 180 costuma ser terra.
        if is_terra:
            # Se for terra, garantimos que a busca eh agressiva
            path, hit_via = ray_march_endpoint(img_hsv, new_surface, sy, sx, dy, dx, base_color, is_terra=True)
            if len(path) > 10: # so aceita se andou um pouco
                # Desenha o caminho
                for (py, px) in path:
                    new_surface[py, px] = 2 # Pinta como terra
                    # Engrossar um pouco
                    cv2.circle(new_surface, (px, py), 2, 2, -1)
                extensions_made += 1
                
        else:
            # Asfalto - tolerancia menor
            path, hit_via = ray_march_endpoint(img_hsv, new_surface, sy, sx, dy, dx, base_color, is_terra=False)
            if len(path) > 10 and hit_via: # Para asfalto, prefere quando conecta em outra
                for (py, px) in path:
                    new_surface[py, px] = 1
                    cv2.circle(new_surface, (px, py), 2, 1, -1)
                extensions_made += 1
                
    print(f">>> Extensoes baseadas na textura: {extensions_made}")
    
    # Conexao por distancia reta (Dijkstra visual) simplificada
    try:
        from skimage.morphology import skeletonize as _sk
        skel_new = _sk(new_surface > 0).astype(np.uint8)
    except ImportError:
        skel_new = (new_surface > 0).astype(np.uint8)
        
    endpoints_new = get_endpoints_and_directions(skel_new, look_back=10)
    
    connections = 0
    used = set()
    for i, ep1 in enumerate(endpoints_new):
        if i in used: continue
        best_j = -1
        best_dist = float('inf')
        for j, ep2 in enumerate(endpoints_new):
            if i == j or j in used: continue
            
            dist = math.hypot(ep1['y']-ep2['y'], ep1['x']-ep2['x'])
            if dist > 200: continue
            
            # Checar alinhamento
            vy, vx = ep2['y']-ep1['y'], ep2['x']-ep1['x']
            norm = math.hypot(vy, vx)
            if norm < 1: continue
            vy, vx = vy/norm, vx/norm
            
            dot1 = ep1['dy']*vy + ep1['dx']*vx
            dot2 = ep2['dy']*(-vy) + ep2['dx']*(-vx)
            ang1 = math.degrees(math.acos(np.clip(dot1, -1.0, 1.0)))
            ang2 = math.degrees(math.acos(np.clip(dot2, -1.0, 1.0)))
            
            if ang1 < 45 and ang2 < 45:
                # Checar assinatura de cor da linha reta
                length = int(dist)
                ys = np.linspace(ep1['y'], ep2['y'], length).astype(int)
                xs = np.linspace(ep1['x'], ep2['x'], length).astype(int)
                
                # Pegar cor nas pontas
                c1 = extract_color_features(img_hsv, ep1['y'], ep1['x'])
                c2 = extract_color_features(img_hsv, ep2['y'], ep2['x'])
                avg_base = (c1 + c2) / 2
                
                # Checar os pontos ao longo da linha
                bad_pixels = 0
                thresh = 0.25
                for yi, xi in zip(ys, xs):
                    c = extract_color_features(img_hsv, yi, xi, radius=1)
                    if color_distance(avg_base, c) > thresh:
                        bad_pixels += 1
                
                if bad_pixels / length < 0.3: # Toleramos ate 30% de pixels diferentes (arvores no caminho)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
                        
        if best_j != -1:
            ep2 = endpoints_new[best_j]
            cv2.line(new_surface, (int(ep1['x']), int(ep1['y'])), (int(ep2['x']), int(ep2['y'])), 2, thickness=3) # Como terra
            used.add(i)
            used.add(best_j)
            connections += 1
            
    print(f">>> Conexoes diretas de lacunas: {connections}")
    
    return new_surface, width_map
