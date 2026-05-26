import numpy as np
import cv2
import math
import heapq

# --- Parâmetros Calibrados ---
HYST_LOW       = 0.04
HYST_HIGH      = 0.30
CLOSE_RADIUS   = 4
MIN_COMPONENT  = 250
ELONGATION_MIN = 2.5       

MAX_BRIDGE_DIST = 180
CONE_ANGLE_DEG  = 50
MIN_LINE_SIGNAL = 0.012

MAX_T_DIST      = 120

DIJKSTRA_MAX_DIST     = 150
DIJKSTRA_MAX_AVG_COST = 11.0
DIJKSTRA_CONE_DEG     = 65
TORTUOSITY_MAX        = 1.8

BORDER_MARGIN = 8

def _make_skel(binary_mask):
    try:
        from skimage.morphology import skeletonize
        return skeletonize(binary_mask > 0).astype(np.uint8)
    except ImportError:
        skel = np.zeros_like(binary_mask)
        img = binary_mask.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        for _ in range(100):
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel

def get_endpoints_and_directions(skel, look_back=10):
    h, w = skel.shape
    k8 = np.ones((3, 3), np.uint8)
    k8[1, 1] = 0
    nbrs = cv2.filter2D(skel, -1, k8)
    
    endpoints = []
    ep_coords = np.argwhere(skel & (nbrs == 1))
    
    for ep in ep_coords:
        sy, sx = int(ep[0]), int(ep[1])
        if sy < BORDER_MARGIN or sy >= h - BORDER_MARGIN or sx < BORDER_MARGIN or sx >= w - BORDER_MARGIN:
            continue # Via truncada
            
        path = [(sy, sx)]
        visited = {(sy, sx)}
        cy, cx = sy, sx
        
        for _ in range(look_back):
            nc = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy==0 and dx==0: continue
                    ny, nx = cy+dy, cx+dx
                    if 0<=ny<h and 0<=nx<w and skel[ny,nx]>0:
                        nc += 1
            if nc >= 3 and len(path) > 1:
                break
                
            found = False
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ny, nx = cy+dy, cx+dx
                if 0<=ny<h and 0<=nx<w and skel[ny,nx]>0 and (ny,nx) not in visited:
                    visited.add((ny,nx))
                    path.append((ny,nx))
                    cy, cx = ny, nx
                    found = True
                    break
            if not found:
                break
                
        if len(path) < 2:
            continue
            
        ey, ex = path[0]
        by, bx = path[-1]
        vy, vx = float(ey - by), float(ex - bx)
        norm = math.hypot(vy, vx)
        if norm >= 0.5:
            endpoints.append({
                'y': sy, 'x': sx,
                'dy': vy/norm, 'dx': vx/norm
            })
            
    return endpoints

def check_line_signal(prob_map, y0, x0, y1, x1):
    length = int(math.hypot(y1-y0, x1-x0))
    if length == 0: return 0.0
    ys = np.linspace(y0, y1, length).astype(int)
    xs = np.linspace(x0, x1, length).astype(int)
    return float(np.mean(prob_map[ys, xs]))

def run(prob_map):
    """
    Funcao principal de bridging. Executa hysteresis, fechamento, PCA de telhados
    e em seguida pontes EE e EL.
    """
    h, w = prob_map.shape
    
    # 1. Hysteresis
    high_mask = prob_map > HYST_HIGH
    low_mask = prob_map > HYST_LOW
    
    num_labels, labels = cv2.connectedComponents(low_mask.astype(np.uint8))
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_labels):
        comp = (labels == i)
        if np.any(comp & high_mask):
            mask[comp] = 1
            
    # 2. Close Gaps
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_RADIUS*2+1, CLOSE_RADIUS*2+1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    
    # 3. Elongation / Filtro anti-telhado
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    filtered = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_COMPONENT:
            continue
            
        comp = (labels == i).astype(np.uint8)
        
        # Se for um componente grande (via legítima), mantem.
        # Telhados gigantes raramente passam de 1000 pixels. 
        # Se usarmos 10000, apagaremos blocos de ruas inteiros que não são tão longos.
        if area > 1000:
            filtered[comp > 0] = 1
            continue
            
        y_pts, x_pts = np.where(comp > 0)
        coords = np.vstack([x_pts, y_pts]).T.astype(np.float32)
        
        if len(coords) < 10:
            filtered[comp > 0] = 1
            continue
            
        mean, evecs, evals = cv2.PCACompute2(coords, mean=None)
        e1 = evals[0,0]
        e2 = evals[1,0] if evals.shape[0] > 1 else 0
        if e2 > 0:
            elongation = np.sqrt(e1 / e2)
        else:
            elongation = float('inf')
            
        if elongation >= ELONGATION_MIN:
            filtered[comp > 0] = 1
            
    mask = filtered
    
    # 4. Bridges (EE e EL) iterativo
    for _iter in range(2):
        skel = _make_skel(mask)
        endpoints = get_endpoints_and_directions(skel)
        bridged = False
        used = set()
        
        # EE Bridge (Endpoint to Endpoint)
        for i, ep1 in enumerate(endpoints):
            if i in used: continue
            best_j = -1
            best_dist = float('inf')
            for j, ep2 in enumerate(endpoints):
                if i == j or j in used: continue
                dist = math.hypot(ep1['y']-ep2['y'], ep1['x']-ep2['x'])
                if dist > MAX_BRIDGE_DIST: continue
                
                v_y, v_x = ep2['y']-ep1['y'], ep2['x']-ep1['x']
                v_norm = math.hypot(v_y, v_x)
                if v_norm < 1: continue
                v_y, v_x = v_y/v_norm, v_x/v_norm
                
                dot1 = ep1['dy']*v_y + ep1['dx']*v_x
                dot2 = ep2['dy']*(-v_y) + ep2['dx']*(-v_x)
                
                angle1 = math.degrees(math.acos(np.clip(dot1, -1.0, 1.0)))
                angle2 = math.degrees(math.acos(np.clip(dot2, -1.0, 1.0)))
                
                if angle1 <= CONE_ANGLE_DEG/2 and angle2 <= CONE_ANGLE_DEG/2:
                    sig = check_line_signal(prob_map, ep1['y'], ep1['x'], ep2['y'], ep2['x'])
                    if sig >= MIN_LINE_SIGNAL:
                        if dist < best_dist:
                            best_dist = dist
                            best_j = j
                            
            if best_j != -1:
                ep2 = endpoints[best_j]
                cv2.line(mask, (ep1['x'], ep1['y']), (ep2['x'], ep2['y']), 1, thickness=2)
                used.add(i)
                used.add(best_j)
                bridged = True
                
        # EL Bridge (T-Junctions)
        for i, ep in enumerate(endpoints):
            if i in used: continue
            ray_y, ray_x = ep['y'], ep['x']
            hit = False
            for step in range(1, MAX_T_DIST):
                ray_y += ep['dy']
                ray_x += ep['dx']
                iy, ix = int(round(ray_y)), int(round(ray_x))
                if not (0 <= iy < h and 0 <= ix < w): break
                
                if mask[iy, ix] > 0 and step > 5:
                    hit = True
                    break
            
            if hit:
                sig = check_line_signal(prob_map, ep['y'], ep['x'], iy, ix)
                if sig >= MIN_LINE_SIGNAL:
                    cv2.line(mask, (ep['x'], ep['y']), (ix, iy), 1, thickness=2)
                    bridged = True
                    used.add(i)
                    
        if not bridged:
            break
            
    # 5. Dijkstra (Caminho de menor custo usando a probabilidade do modelo)
    # Constroi um mapa de custo
    cost_map = 1.0 / (prob_map + 0.001)
    
    skel = _make_skel(mask)
    endpoints = get_endpoints_and_directions(skel)
    for ep in endpoints:
        # Pula se ja esta bem conectado
        if mask[ep['y'], ep['x']] == 0: continue
        
        # A* simplificado
        pq = [(0, ep['y'], ep['x'], [])] # cost, y, x, path
        visited = set()
        best_path = None
        best_cost_avg = float('inf')
        
        while pq:
            c, cy, cx, path = heapq.heappop(pq)
            if (cy, cx) in visited: continue
            visited.add((cy, cx))
            
            # Só consideramos que encostou em outra rua se tivermos andado
            # uma distância mínima para fora do "grosso" da rua de onde saímos.
            if len(path) > 8 and mask[cy, cx] > 0:
                dist = len(path)
                avg_cost = c / dist
                if dist <= DIJKSTRA_MAX_DIST and avg_cost <= DIJKSTRA_MAX_AVG_COST:
                    tortuosity = dist / max(1, math.hypot(cy-ep['y'], cx-ep['x']))
                    if tortuosity <= TORTUOSITY_MAX:
                        best_path = path
                        best_cost_avg = avg_cost
                break
                
            if len(path) > DIJKSTRA_MAX_DIST:
                continue
                
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                ny, nx = cy+dy, cx+dx
                if 0<=ny<h and 0<=nx<w and (ny, nx) not in visited:
                    # Direcao do cone
                    vy, vx = ny-ep['y'], nx-ep['x']
                    norm = math.hypot(vy, vx)
                    if norm > 0:
                        dot = ep['dy']*(vy/norm) + ep['dx']*(vx/norm)
                        ang = math.degrees(math.acos(np.clip(dot, -1.0, 1.0)))
                        if ang <= DIJKSTRA_CONE_DEG/2:
                            nc = c + cost_map[ny, nx]
                            heapq.heappush(pq, (nc, ny, nx, path + [(ny, nx)]))
                            
        if best_path:
            for py, px in best_path:
                mask[py, px] = 1
                
    return mask * 255
