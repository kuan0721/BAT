# astar_spc_critical_pseudo_anchored_splits.py
from heapq import heappush, heappop
from math import inf
from collections import deque

# ----- Grid (Cartesian): (0,0)=bottom-left, (6,7)=top-right -----
R, C = 8, 7
DIRS_XY = [(0,1),(1,0),(0,-1),(-1,0)]  # 4-connected

# --- Coordinate conversions -------------------------------------------------
def xy_to_rc(x, y):  return (R - 1 - y, x)
def rc_to_xy(r, c):  return (c, R - 1 - r)
def in_bounds_xy(x, y):  return 0 <= x < C and 0 <= y < R
def h_manhattan_xy(x1, y1, x2, y2):  return abs(x1-x2)+abs(y1-y2)

# --- Grid builder -----------------------------------------------------------
def build_grid_from_obstacles_xy(obstacles_xy):
    grid = [[0 for _ in range(C)] for _ in range(R)]
    for (x, y) in obstacles_xy:
        if in_bounds_xy(x, y):
            r, c = xy_to_rc(x, y)
            grid[r][c] = 1
    return grid

# --- A* (4-connected, Cartesian) -------------------------------------------
def astar_cartesian(obstacles_xy, start_xy=(0,0), goal_xy=(C-1,R-1)):
    grid = build_grid_from_obstacles_xy(obstacles_xy)
    sx, sy = start_xy; gx, gy = goal_xy
    if not in_bounds_xy(sx, sy) or not in_bounds_xy(gx, gy): return [], inf
    sr, sc = xy_to_rc(sx, sy); gr, gc = xy_to_rc(gx, gy)
    if grid[sr][sc]==1 or grid[gr][gc]==1: return [], inf

    g = [[inf]*C for _ in range(R)]
    parent = [[(-1,-1) for _ in range(C)] for _ in range(R)]
    closed = [[False]*C for _ in range(R)]
    pq = []
    g[sr][sc]=0.0
    heappush(pq,(h_manhattan_xy(sx,sy,gx,gy),0.0,sr,sc,-1,-1))

    while pq:
        f,cg,r,c,pr,pc = heappop(pq)
        if closed[r][c]: continue
        closed[r][c]=True
        parent[r][c]=(pr,pc)
        if (r,c)==(gr,gc):
            path=[]
            cr,cc=r,c
            while cr!=-1 and cc!=-1:
                path.append(rc_to_xy(cr,cc))
                cr,cc=parent[cr][cc]
            return list(reversed(path)), g[r][c]
        cx,cy=rc_to_xy(r,c)
        for dx,dy in DIRS_XY:
            nx,ny=cx+dx,cy+dy
            if not in_bounds_xy(nx,ny): continue
            nr,nc=xy_to_rc(nx,ny)
            if grid[nr][nc]==1 or closed[nr][nc]: continue
            tg=g[r][c]+1
            if tg<g[nr][nc]:
                g[nr][nc]=tg
                nf=tg+h_manhattan_xy(nx,ny,gx,gy)
                heappush(pq,(nf,tg,nr,nc,r,c))
    return [],inf

# --- Helpers ----------------------------------------------------------------
def _neighbors4_xy(x, y):
    for dx,dy in DIRS_XY:
        nx,ny=x+dx,y+dy
        if in_bounds_xy(nx,ny): yield (nx,ny)

def _free_neighbors4_xy(x,y,grid):
    for dx,dy in DIRS_XY:
        nx,ny=x+dx,y+dy
        if in_bounds_xy(nx,ny):
            r,c=xy_to_rc(nx,ny)
            if grid[r][c]==0: yield (nx,ny)

# --- Manhattan δ-SPC (obstacle-aware BFS from path) -------------------------
def SPC_manhattan_blocked(obstacles_xy,d,start_xy=(0,0),goal_xy=None):
    if goal_xy is None: goal_xy=(C-1,R-1)
    grid=build_grid_from_obstacles_xy(obstacles_xy)
    path,_=astar_cartesian(obstacles_xy,start_xy,goal_xy)
    if not path: return set(),[]
    spc=set()
    for (px,py) in path:
        q=deque([(px,py,0)])
        seen={(px,py)}
        while q:
            x,y,dist=q.popleft()
            spc.add((x,y))
            if dist==d: continue
            for nx,ny in _free_neighbors4_xy(x,y,grid):
                if (nx,ny) not in seen:
                    seen.add((nx,ny))
                    q.append((nx,ny,dist+1))
    return spc,path

# --- Pruning redundant SPC cells -------------------------------------------
def redundant_cells_with_degree_one(spc_xy, obstacles_xy, path_xy=(), exclude_endpoints=True):
    spc=set(spc_xy)
    protected=set(path_xy)
    if exclude_endpoints and path_xy:
        protected.add(path_xy[0]); protected.add(path_xy[-1])
    red=set()
    for cell in spc:
        if cell in protected: continue
        deg=sum((nbr in spc) for nbr in _neighbors4_xy(*cell))
        if deg==1: red.add(cell)
    return red

def prune_redundant_cells_iteratively(spc_xy, obstacles_xy, path_xy=(), exclude_endpoints=True):
    spc=set(spc_xy); removed_all=set()
    protected=set(path_xy)
    if exclude_endpoints and path_xy:
        protected.add(path_xy[0]); protected.add(path_xy[-1])
    while True:
        red=redundant_cells_with_degree_one(spc, obstacles_xy, path_xy, exclude_endpoints)
        if not red: break
        red-=protected
        spc-=red
        removed_all|=red
    return spc, removed_all

def redundant_cells_with_degree_two(spc_xy, obstacles_xy, path_xy=(), exclude_endpoints=True):
    spc=set(spc_xy); removed_all=set()
    protected=set(path_xy)
    if exclude_endpoints and path_xy:
        protected.add(path_xy[0]); protected.add(path_xy[-1])
    while True:
        degree={cell:sum((nbr in spc) for nbr in _neighbors4_xy(*cell)) for cell in spc}
        to_remove=set()
        for u in spc:
            if u in protected or degree[u]!=2: continue
            for v in _neighbors4_xy(*u):
                if v in spc and v not in protected and degree.get(v,0)==2:
                    to_remove.add(u); to_remove.add(v); break
        if not to_remove: break
        to_remove-=protected
        spc-=to_remove
        removed_all|=to_remove
    return spc, removed_all

# --- Critical & Pseudo-critical (STRICT as requested) -----------------------
def critical_cells_in_spc(spc_xy, path_xy):
    """
    Critical (c): path cell satisfying:
      1) SPC degree <= 2
      2) All SPC neighbors are path cells
      3) Among those SPC neighbors, at most one has SPC degree >= 3
    """
    spc_set  = set(spc_xy)
    path_set = set(path_xy)
    deg_map = {cell: sum((nbr in spc_set) for nbr in _neighbors4_xy(*cell)) for cell in spc_set}
    critical = set()
    for cell in path_xy:
        if cell not in spc_set: continue
        nbrs = [nbr for nbr in _neighbors4_xy(*cell) if nbr in spc_set]
        d_self = deg_map.get(cell, 0)
        if d_self > 2: continue
        if not all(n in path_set for n in nbrs): continue
        high_deg_nbrs = sum(1 for n in nbrs if deg_map.get(n, 0) >= 3)
        if high_deg_nbrs <= 1:
            critical.add(cell)
    return critical

def pseudo_critical_cells_in_spc(spc_xy, path_xy, critical_xy, *, exclude_critical_self=True):
    crit = set(critical_xy)
    pseudo = set()
    for cell in path_xy:
        if exclude_critical_self and cell in crit:  # don't double-count
            continue
        x, y = cell
        if any(((x+dx, y+dy) in crit) for dx,dy in DIRS_XY):
            pseudo.add(cell)
    return pseudo

# --- Rendering --------------------------------------------------------------
def grid_with_symbols(obstacles_xy, spc_xy, path_xy, marks=None):
    grid = build_grid_from_obstacles_xy(obstacles_xy)
    disp = [['#' if grid[r][c] == 1 else '.' for c in range(C)] for r in range(R)]
    for (x, y) in spc_xy:
        r, c = xy_to_rc(x, y)
        if disp[r][c] == '.': disp[r][c] = 'o'
    for (x, y) in path_xy:
        r, c = xy_to_rc(x, y)
        if disp[r][c] != '#': disp[r][c] = '*'
    if path_xy:
        sx, sy = path_xy[0]; gx, gy = path_xy[-1]
        rs, cs = xy_to_rc(sx, sy); rg, cg = xy_to_rc(gx, gy)
        disp[rs][cs] = 'S'; disp[rg][cg] = 'G'
    if marks:
        for (x, y), ch in marks.items():
            r, c = xy_to_rc(x, y)
            if disp[r][c] not in ('#','S','G'): disp[r][c] = ch
    return disp

def print_matrix(mat, title=None):
    if title: print(title)
    for row in mat: print(' '.join(row))
    print()

# --- Anchored split-by-pseudo reporting -------------------------------------
def _split_sequence_by_pseudo_anchored(sorted_vals, decide_fn):
    """
    Split a sorted 1D sequence into segments with direction control:
      decide_fn(v) -> 'before' | 'after' | None
        - 'after'  : include v in current segment, then break (split after v)
        - 'before' : break BEFORE v (current segment ends), start new with [v]
        - None     : no split at v; just append to current
    """
    segs = []
    current = []
    for v in sorted_vals:
        decision = decide_fn(v)
        if decision == 'before':
            if current:
                segs.append(current)
            current = [v]
        elif decision == 'after':
            current.append(v)
            segs.append(current)
            current = []
        else:  # None
            current.append(v)
    if current:
        segs.append(current)
    return segs

def print_spc_by_rowcol_split_on_pseudo_anchored(spc_xy, critical_xy, pseudo_xy):
    """
    Print SPC cells grouped by row and column, splitting at pseudo cells,
    with each pseudo grouped with an adjacent critical cell's segment:
      - Rows (y fixed): prefers LEFT critical; otherwise RIGHT; else defaults to 'after'
      - Columns (x fixed): prefers LOWER critical; otherwise UPPER; else defaults to 'after'
    """
    if not spc_xy:
        print("(Empty SPC)\n")
        return

    spc = set(spc_xy)
    crit = set(critical_xy)
    pseudo = set(pseudo_xy)

    # Rows (y fixed)
    rows = {}
    for (x, y) in spc:
        rows.setdefault(y, []).append(x)
    print("Cells by row (y), split on pseudo and anchored to adjacent critical:")
    for y in sorted(rows.keys()):
        xs = sorted(rows[y])
        def decide_row(x):
            if (x, y) not in pseudo:   # regular SPC cell
                return None
            # Anchor to adjacent critical on row if present
            has_left  = (x-1, y) in crit
            has_right = (x+1, y) in crit
            if has_left:
                return 'after'   # include (x,y) in LEFT segment
            if has_right:
                return 'before'  # start RIGHT segment at (x,y)
            return 'after'       # default behavior
        segs = _split_sequence_by_pseudo_anchored(xs, decide_row)
        for idx, seg in enumerate(segs):
            print(f"  Row {y:02d}-{idx:02d}: {seg}")
    print()

    # Columns (x fixed)
    cols = {}
    for (x, y) in spc:
        cols.setdefault(x, []).append(y)
    print("Cells by column (x), split on pseudo and anchored to adjacent critical:")
    for x in sorted(cols.keys()):
        ys = sorted(cols[x])
        def decide_col(y):
            if (x, y) not in pseudo:
                return None
            # Anchor to adjacent critical on column if present
            has_down = (x, y-1) in crit  # lower neighbor
            has_up   = (x, y+1) in crit  # upper neighbor
            if has_down:
                return 'after'   # include (x,y) in LOWER segment
            if has_up:
                return 'before'  # start UPPER segment at (x,y)
            return 'after'       # default behavior
        segs = _split_sequence_by_pseudo_anchored(ys, decide_col)
        for idx, seg in enumerate(segs):
            print(f"  Col {x:02d}-{idx:02d}: {seg}")
    print()

# --- Demo -------------------------------------------------------------------
if __name__=="__main__":
    obstacles_xy={
        (1,0),(4,1),(5,1),(0,4),(4,4),(3,4),(6,4),
        (2,5),(4,5),(6,5),
        (0,6),(1,6),(1,7),(2,7),(4,7),(5,7)
    }
    start_xy=(0,0); goal_xy=(C-1,R-1)

    path_xy, cost = astar_cartesian(obstacles_xy, start_xy, goal_xy)
    if not path_xy:
        print("No path found.")
    else:
        print("A* path:", path_xy, "\n")

        for d in (1,2,3):
            print(f"==== δ = {d} ====")
            spc_xy, _ = SPC_manhattan_blocked(obstacles_xy, d, start_xy, goal_xy)
            spc_after_deg1, _ = prune_redundant_cells_iteratively(spc_xy, obstacles_xy, path_xy)
            spc_final, _ = redundant_cells_with_degree_two(spc_after_deg1, obstacles_xy, path_xy)

            # Maps
            mat_final = grid_with_symbols(obstacles_xy, spc_final, path_xy)
            print_matrix(mat_final, "Final pruned SPC (o) with path")

            # Critical & pseudo-critical
            critical_xy = critical_cells_in_spc(spc_final, path_xy)
            pseudo_xy   = pseudo_critical_cells_in_spc(spc_final, path_xy, critical_xy, exclude_critical_self=True)

            # Overlays
            mat_c = grid_with_symbols(obstacles_xy, spc_final, path_xy, {c:'c' for c in critical_xy})
            mat_p = grid_with_symbols(obstacles_xy, spc_final, path_xy, {p:'p' for p in pseudo_xy})
            print_matrix(mat_c, "Critical cells marked 'c'")
            print_matrix(mat_p, "Pseudo-critical cells marked 'p'")

            print(f"Critical cells (c):        {sorted(critical_xy)}")
            print(f"Pseudo-critical cells (p): {sorted(pseudo_xy)}\n")

            # NEW: anchored splits by row/column using adjacency to criticals
            print_spc_by_rowcol_split_on_pseudo_anchored(spc_final, critical_xy, pseudo_xy)
