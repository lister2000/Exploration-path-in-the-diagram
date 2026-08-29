#!/usr/bin/env python3
"""
交互式区域生长 + 最长链条追踪工具（平滑路径版 + 结果缩放 + 空格重来）
========================================================================
操作流程：
  第一步：鼠标右键拖框 → 放大图片；左键拖框 → 定义样本区域
  第二步：自动读取样本区域灰度均值和方差
  第三步：按样本区域大小把全图划分网格
  第四步：从样本所在网格开始，扩展到周围邻居
  第五步：只保留灰度均值和方差与样本接近的格子
  第六步：DFS 继续扩展，直到没有新的相似格子
  第七步：从所有路径中，智能评分选择最优路径
  第八步：绘制平滑路径

结果窗口操作：
  - 左键单击: 恢复原始视图
  - 右键拖框: 放大查看细节（矩形框实时显示）
  - 空格键: 重新选择样本 (重来)
  - R键: 恢复原始视图
  - Z键: 回退上一级缩放
  - S键: 保存截图
  - ESC键: 退出程序

依赖：pip install opencv-python numpy scipy
使用：python region_growing_tool.py [图片路径]    （默认 test.jpg）
"""

import cv2
import numpy as np
import sys
import os
import time

# ======================== 用户可配置参数 ========================

# ---- 路径平滑配置 ----
ENABLE_PATH_SMOOTHING = True   # 是否启用路径平滑
SMOOTHING_FACTOR = 0.5         # 平滑因子 (0-1)，越大越平滑
PATH_LINE_WIDTH = 4            # 路径线宽
PATH_POINT_RADIUS = 5          # 路径节点半径
SHOW_GRID_LINES = False        # 是否显示网格线

# ---- 路径评分权重 ----
WEIGHT_LENGTH = 1.0            # 路径长度权重
WEIGHT_SMOOTHNESS = 0.5        # 路径平滑度权重
WEIGHT_CONSISTENCY = 0.3       # 路径一致性权重

# ---- 显示信息配置 ----
DISPLAY_INFO = [
    'filter',
    'factor',
    'neighbor',
    'mean',
    'std',
    'threshold',
    'grown',
    'chain',
    'paths',
    'score',
]

# ---- 邻域配置 ----
USE_8_NEIGHBOR = True          # True: 8邻域, False: 4邻域

# ---- 文字显示配置 ----
TEXT_SCALE = 1.6
TEXT_THICKNESS = 3
TEXT_LINE_SPACING = 50
TEXT_START_Y = 45
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)
TEXT_BG_ALPHA = 0.6

# ---- 采样预览配置 ----
SAMPLE_PREVIEW_SIZE = 300
SAMPLE_PREVIEW_SHOW = True

# ---- 算法参数 ----
DEFAULT_IMAGE = "test.jpg"
MIN_STD = 10
MAX_DISPLAY = 1200

# ======================== 全局变量 ========================
current_image = None
current_sample_rect = None
current_result = None
current_chain = []
slider_filter = 150
slider_factor = 2.0

# ======================== 路径平滑工具 ========================

def smooth_path_with_spline(points, num_points=None, smoothing_factor=0.5):
    """使用B样条插值平滑路径"""
    if len(points) < 3:
        return points
    
    pts = np.array(points)
    
    try:
        from scipy.interpolate import splprep, splev
        
        is_closed = False
        if np.linalg.norm(pts[0] - pts[-1]) < 10:
            is_closed = True
        
        if is_closed:
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=smoothing_factor * 10, per=True)
        else:
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=smoothing_factor * 10)
        
        if num_points is None:
            num_points = len(points) * 3
        
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        
        smooth_points = list(zip(x_new.astype(int), y_new.astype(int)))
        return smooth_points
        
    except Exception as e:
        print(f"[平滑] 警告: 样条插值失败 ({e})，使用原始点")
        return points

def get_subpixel_path_points(chain, cell_w, cell_h, gray_image):
    """在格子内部使用亚像素定位"""
    points = []
    
    for r, c in chain:
        cx = c * cell_w + cell_w // 2
        cy = r * cell_h + cell_h // 2
        
        x1 = max(0, c * cell_w)
        y1 = max(0, r * cell_h)
        x2 = min(gray_image.shape[1], (c + 1) * cell_w)
        y2 = min(gray_image.shape[0], (r + 1) * cell_h)
        
        patch = gray_image[y1:y2, x1:x2]
        
        if patch.size > 0:
            h_patch, w_patch = patch.shape
            yy, xx = np.mgrid[:h_patch, :w_patch]
            
            weights = patch.astype(np.float32)
            if weights.sum() > 0:
                weights = weights / (weights.sum() + 1e-6)
                
                cy_sub = (yy * weights).sum()
                cx_sub = (xx * weights).sum()
                
                sub_x = int(x1 + cx_sub)
                sub_y = int(y1 + cy_sub)
                
                if (sub_x >= 0 and sub_x < gray_image.shape[1] and 
                    sub_y >= 0 and sub_y < gray_image.shape[0]):
                    points.append((sub_x, sub_y))
                else:
                    points.append((cx, cy))
            else:
                points.append((cx, cy))
        else:
            points.append((cx, cy))
    
    return points

def draw_smooth_path(img, points, color=(0, 0, 255), width=4, 
                     show_nodes=True, node_radius=5, 
                     show_gradient=True):
    """绘制平滑路径"""
    if len(points) < 2:
        return img
    
    result = img.copy()
    
    if show_gradient and len(points) > 2:
        for i in range(len(points) - 1):
            ratio = i / (len(points) - 1)
            r = int(255 * ratio)
            g = int(200 * (1 - ratio))
            b = int(200 * (1 - ratio))
            color_step = (b, g, r)
            cv2.line(result, points[i], points[i + 1], color_step, width, cv2.LINE_AA)
    else:
        for i in range(len(points) - 1):
            cv2.line(result, points[i], points[i + 1], color, width, cv2.LINE_AA)
    
    if show_nodes:
        for i, pt in enumerate(points):
            if i == 0:
                cv2.circle(result, pt, node_radius + 3, (0, 255, 255), -1)
                cv2.circle(result, pt, node_radius + 3, (0, 0, 0), 1)
                cv2.putText(result, "S", (pt[0] - 10, pt[1] + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            elif i == len(points) - 1:
                cv2.circle(result, pt, node_radius + 3, (0, 0, 255), -1)
                cv2.circle(result, pt, node_radius + 3, (0, 0, 0), 1)
                cv2.putText(result, "E", (pt[0] - 10, pt[1] + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.circle(result, pt, node_radius, (0, 0, 255), -1)
                cv2.circle(result, pt, node_radius, (255, 255, 255), 1)
    
    return result

# ======================== 工具函数 ========================

def fit_size(img, max_side=MAX_DISPLAY):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = img.copy()
    return small, img, scale

def grid_mean_grid(gray, cell_h, cell_w, filter_threshold):
    gh = gray.shape[0] // cell_h
    gw = gray.shape[1] // cell_w
    means = np.zeros((gh, gw), dtype=np.float64)
    valid_count = np.zeros((gh, gw), dtype=np.int32)
    
    for r in range(gh):
        for c in range(gw):
            patch = gray[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            masked = patch[patch <= filter_threshold]
            valid_count[r, c] = len(masked)
            if len(masked) > 0:
                means[r, c] = masked.mean()
            else:
                means[r, c] = 999
    return means, valid_count

def calculate_path_smoothness(path, grid_means):
    if len(path) < 2:
        return 0.0
    
    total_diff = 0.0
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        diff = abs(grid_means[r1, c1] - grid_means[r2, c2])
        total_diff += diff
    
    return total_diff / (len(path) - 1)

def calculate_path_consistency(path, grid_means, reference_mean):
    if not path:
        return 0.0
    
    total_diff = 0.0
    for r, c in path:
        diff = abs(grid_means[r, c] - reference_mean)
        total_diff += diff
    
    return total_diff / len(path)

def calculate_path_score(path, grid_means, reference_mean, weights=None):
    if weights is None:
        weights = {
            'length': WEIGHT_LENGTH,
            'smoothness': WEIGHT_SMOOTHNESS,
            'consistency': WEIGHT_CONSISTENCY
        }
    
    if len(path) < 2:
        return 0.0, {'length': 0, 'smoothness': 0, 'consistency': 0}
    
    length_score = min(len(path) / 100.0, 1.0)
    
    smoothness_raw = calculate_path_smoothness(path, grid_means)
    smoothness_score = max(1.0 - smoothness_raw / 50.0, 0.0)
    
    consistency_raw = calculate_path_consistency(path, grid_means, reference_mean)
    consistency_score = max(1.0 - consistency_raw / 50.0, 0.0)
    
    total_score = (
        weights['length'] * length_score +
        weights['smoothness'] * smoothness_score +
        weights['consistency'] * consistency_score
    ) / sum(weights.values())
    
    details = {
        'length': length_score,
        'smoothness': smoothness_score,
        'consistency': consistency_score,
        'length_raw': len(path),
        'smoothness_raw': smoothness_raw,
        'consistency_raw': consistency_raw,
        'total': total_score
    }
    
    return total_score, details

def find_all_paths_dfs(grid_means, start_keys, reference_mean, threshold, grid_h, grid_w, use_8_neighbor=True):
    if use_8_neighbor:
        dirs = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)]
    else:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    parent = {}
    depth = {}
    visited = set()
    all_visited = set()
    
    for key in start_keys:
        if grid_means[key[0], key[1]] < 900:
            parent[key] = None
            depth[key] = 0
            visited.add(key)
            all_visited.add(key)
    
    if not visited:
        return [], set(), {}
    
    stack = list(visited)
    
    while stack:
        current = stack.pop()
        
        for dr, dc in dirs:
            nr, nc = current[0] + dr, current[1] + dc
            
            if not (0 <= nr < grid_h and 0 <= nc < grid_w):
                continue
            
            neighbor = (nr, nc)
            
            if neighbor in visited:
                continue
            
            neighbor_mean = grid_means[nr, nc]
            if neighbor_mean >= 900:
                continue
            
            diff = abs(neighbor_mean - reference_mean)
            if diff >= threshold:
                continue
            
            visited.add(neighbor)
            all_visited.add(neighbor)
            parent[neighbor] = current
            depth[neighbor] = depth[current] + 1
            stack.append(neighbor)
    
    children_count = {}
    for node, p in parent.items():
        if p is not None:
            children_count[p] = children_count.get(p, 0) + 1
    
    leaves = []
    for node in visited:
        if node not in children_count:
            leaves.append(node)
    
    all_paths = []
    for leaf in leaves:
        path = []
        current = leaf
        while current is not None:
            path.append(current)
            current = parent.get(current)
        path.reverse()
        all_paths.append(path)
    
    return all_paths, all_visited, depth

def score_and_select_best_path(all_paths, grid_means, reference_mean):
    if not all_paths:
        return [], {}, None
    
    scored_paths = []
    for path in all_paths:
        score, details = calculate_path_score(path, grid_means, reference_mean)
        scored_paths.append({
            'path': path,
            'score': score,
            'details': details
        })
    
    scored_paths.sort(key=lambda x: x['score'], reverse=True)
    
    best_item = scored_paths[0]
    best_path = best_item['path']
    best_score = best_item['score']
    best_details = best_item['details']
    
    return best_path, best_details, scored_paths

# ======================== 显示函数 ========================

def print_sample_info(sample_rect, sample_mean, sample_std, filter_threshold):
    sx1, sy1, sx2, sy2 = sample_rect
    sample_w = sx2 - sx1
    sample_h = sy2 - sy1
    
    print("\n" + "=" * 60)
    print("📊 采样信息")
    print("=" * 60)
    print(f"  样本区域: ({sx1}, {sy1}) - ({sx2}, {sy2})")
    print(f"  样本尺寸: {sample_w} x {sample_h} 像素")
    print(f"  灰度均值: {sample_mean:.2f}")
    print(f"  灰度标准差: {sample_std:.2f}")
    print(f"  过滤阈值: {filter_threshold} (排除 > {filter_threshold} 的像素)")
    print("=" * 60)

def show_sample_preview(image, sample_rect, filter_threshold, return_canvas=False):
    """
    显示采样区域的放大预览窗口（修复版）

    参数:
        return_canvas: True 时不打开窗口，只返回画布数组，便于单元测试与脚本集成
    """
    sx1, sy1, sx2, sy2 = sample_rect
    sample_crop = image[sy1:sy2, sx1:sx2].copy()
    
    if sample_crop.size == 0:
        print("[警告] 采样区域为空，无法预览")
        return None
    
    h, w = sample_crop.shape[:2]
    target_size = SAMPLE_PREVIEW_SIZE
    
    # 计算缩放比例，保持宽高比
    scale = min(target_size / h, target_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 确保 new_w 和 new_h 至少为 1
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    # 缩放图像 - 注意 cv2.resize 的尺寸参数是 (width, height)
    resized = cv2.resize(sample_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 创建画布（目标大小）
    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 128
    
    # 计算居中位置
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    
    # 确保偏移量非负
    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)
    
    # 确保切片范围不超过画布边界
    y_end = min(y_offset + new_h, target_size)
    x_end = min(x_offset + new_w, target_size)
    
    # 将缩放后的图像放到画布中央
    canvas[y_offset:y_end, x_offset:x_end] = resized[:y_end-y_offset, :x_end-x_offset]
    
    # ---- 添加信息标注 ----
    gray = cv2.cvtColor(sample_crop, cv2.COLOR_BGR2GRAY)
    masked = gray[gray <= filter_threshold]
    if len(masked) > 0:
        mean_val = masked.mean()
        std_val = masked.std()
    else:
        mean_val = 0
        std_val = 0
    
    info_lines = [
        f"Sample Preview",
        f"Size: {w}x{h}  Mean: {mean_val:.1f}  Std: {std_val:.1f}",
        f"Filter: {filter_threshold}  Valid: {len(masked)}",
    ]
    
    y_pos = target_size - 10
    for line in reversed(info_lines):
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (5, y_pos - th - 5), (tw + 15, y_pos + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        cv2.putText(canvas, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        y_pos -= (th + 8)
    
    cv2.rectangle(canvas, (0, 0), (target_size - 1, target_size - 1), (0, 255, 0), 2)
    
    if return_canvas:
        return canvas

    window_name = "Sample Preview (Press any key to continue)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, target_size, target_size)
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return canvas

def get_info_lines(stats, sample_rect=None):
    info_map = {
        'filter': f"Filter: {stats.get('filter_threshold', 'N/A')}",
        'factor': f"Factor: {stats.get('factor', 'N/A'):.1f}",
        'neighbor': f"Neighbor: {'8' if USE_8_NEIGHBOR else '4'}",
        'mean': f"Mean: {stats.get('sample_mean', 'N/A'):.1f}",
        'std': f"Std: {stats.get('sample_std', 'N/A'):.1f}",
        'cell_size': f"Grid: {stats.get('cell_w', 'N/A')}x{stats.get('cell_h', 'N/A')}",
        'grid_count': f"Cells: {stats.get('grid_w', 'N/A')}x{stats.get('grid_h', 'N/A')}",
        'threshold': f"Thresh: {stats.get('threshold', 'N/A'):.1f}",
        'grown': f"Nodes: {stats.get('grown_count', 'N/A')}",
        'chain': f"Chain: {stats.get('chain_length', 'N/A')}",
        'paths': f"Paths: {stats.get('path_count', 'N/A')}",
        'score': f"Score: {stats.get('score', 'N/A'):.3f}",
        'sample_rect': f"Rect: {sample_rect}" if sample_rect else "",
    }
    
    if 'all' in DISPLAY_INFO:
        keys_to_show = list(info_map.keys())
    else:
        keys_to_show = DISPLAY_INFO
    
    lines = []
    for key in keys_to_show:
        if key in info_map and info_map[key]:
            lines.append(info_map[key])
    
    return lines

def draw_info_text(img, info_lines, config=None):
    if config is None:
        config = {}
    
    scale = config.get('scale', TEXT_SCALE)
    thickness = config.get('thickness', TEXT_THICKNESS)
    line_spacing = config.get('line_spacing', TEXT_LINE_SPACING)
    start_y = config.get('start_y', TEXT_START_Y)
    color = config.get('color', TEXT_COLOR)
    bg_color = config.get('bg_color', TEXT_BG_COLOR)
    bg_alpha = config.get('bg_alpha', TEXT_BG_ALPHA)
    
    if not info_lines:
        return
    
    max_width = 0
    for line in info_lines:
        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        max_width = max(max_width, w)
    
    total_height = len(info_lines) * line_spacing + 20
    
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (max_width + 25, total_height + 10), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    
    y_pos = start_y
    for line in info_lines:
        cv2.putText(img, line, (12, y_pos + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 1)
        cv2.putText(img, line, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness)
        y_pos += line_spacing

# ======================== 核心算法 ========================

def region_growing_with_params(image, sample_rect, filter_threshold, factor, show_preview=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sx1, sy1, sx2, sy2 = sample_rect

    # ---- 样本统计 ----
    sample_patch = gray[sy1:sy2, sx1:sx2]
    sample_masked = sample_patch[sample_patch <= filter_threshold]
    
    if len(sample_masked) == 0:
        print(f"[错误] 样本区域内没有 ≤ {filter_threshold} 的像素")
        return image.copy(), [], None
    
    reference_mean = float(sample_masked.mean())
    reference_std = float(sample_masked.std())
    
    print_sample_info(sample_rect, reference_mean, reference_std, filter_threshold)
    
    if show_preview and SAMPLE_PREVIEW_SHOW:
        show_sample_preview(image, sample_rect, filter_threshold)
    
    # ---- 划网格 ----
    cell_h = max(sy2 - sy1, 1)
    cell_w = max(sx2 - sx1, 1)
    grid_h = h // cell_h
    grid_w = w // cell_w
    
    print(f"\n[网格] 尺寸: {cell_w}x{cell_h}, 数量: {grid_w}x{grid_h}")
    
    grid_means, valid_counts = grid_mean_grid(gray, cell_h, cell_w, filter_threshold)

    # ---- 找到样本覆盖的所有网格作为起点 ----
    sr1 = max(0, sy1 // cell_h - 1)
    sc1 = max(0, sx1 // cell_w - 1)
    sr2 = min(grid_h - 1, (sy2 - 1) // cell_h + 1)
    sc2 = min(grid_w - 1, (sx2 - 1) // cell_w + 1)
    
    start_keys = []
    for r in range(sr1, sr2 + 1):
        for c in range(sc1, sc2 + 1):
            if grid_means[r, c] < 900:
                start_keys.append((r, c))
    
    if not start_keys:
        print("[错误] 没有有效的起点")
        return image.copy(), [], None
    
    start_keys.sort(key=lambda k: valid_counts[k[0], k[1]], reverse=True)
    
    print(f"[起点] 找到 {len(start_keys)} 个有效起点")

    # ---- 计算相似度阈值 ----
    threshold = max(factor * reference_std, MIN_STD)
    print(f"[相似度阈值] {threshold:.2f}")

    # ---- 全遍历找所有路径 ----
    all_paths, all_visited, depth = find_all_paths_dfs(
        grid_means, start_keys, reference_mean, threshold, grid_h, grid_w, USE_8_NEIGHBOR
    )
    
    if not all_paths:
        print("[错误] 没有找到任何路径")
        return image.copy(), [], None
    
    # ---- 智能评分选择最优路径 ----
    best_path, best_details, scored_paths = score_and_select_best_path(
        all_paths, grid_means, reference_mean
    )
    
    if not best_path:
        print("[错误] 无法选择最优路径")
        return image.copy(), [], None
    
    chain = best_path
    
    # ---- 统计信息 ----
    stats = {
        'sample_mean': reference_mean,
        'sample_std': reference_std,
        'threshold': threshold,
        'cell_h': cell_h,
        'cell_w': cell_w,
        'grid_h': grid_h,
        'grid_w': grid_w,
        'grown_count': len(all_visited),
        'chain_length': len(chain),
        'path_count': len(all_paths),
        'score': best_details['total'],
        'filter_threshold': filter_threshold,
        'factor': factor,
    }

    # ---- 绘制结果 ----
    result = image.copy()

    # 1. 先用半透明白色填充所有网格
    if SHOW_GRID_LINES:
        white_overlay = result.copy()
        for r in range(grid_h):
            for c in range(grid_w):
                y1 = r * cell_h
                x1 = c * cell_w
                cv2.rectangle(white_overlay, (x1, y1), (x1 + cell_w, y1 + cell_h), (255, 255, 255), -1)
        cv2.addWeighted(white_overlay, 0.2, result, 0.8, 0, result)

    # 2. 画所有被访问过的节点
    visited_without_path = all_visited - set(chain)
    if visited_without_path:
        overlay_visited = result.copy()
        for (r, c) in visited_without_path:
            y1 = r * cell_h
            x1 = c * cell_w
            cv2.rectangle(overlay_visited, (x1, y1), (x1 + cell_w, y1 + cell_h), (150, 180, 220), -1)
        cv2.addWeighted(overlay_visited, 0.2, result, 0.8, 0, result)

    # 3. 画路径上的格子
    overlay_path = result.copy()
    for (r, c) in chain:
        y1 = r * cell_h
        x1 = c * cell_w
        cv2.rectangle(overlay_path, (x1, y1), (x1 + cell_w, y1 + cell_h), (200, 200, 100), -1)
    cv2.addWeighted(overlay_path, 0.2, result, 0.8, 0, result)

    # 4. 画网格线
    if SHOW_GRID_LINES:
        for r in range(grid_h + 1):
            y = min(r * cell_h, h - 1)
            cv2.line(result, (0, y), (w, y), (200, 200, 200), 1)
        for c in range(grid_w + 1):
            x = min(c * cell_w, w - 1)
            cv2.line(result, (x, 0), (x, h), (200, 200, 200), 1)

    # 5. 画样本区域
    cv2.rectangle(result, (sx1, sy1), (sx2, sy2), (0, 255, 0), 3)

    # 6. 绘制平滑路径
    if ENABLE_PATH_SMOOTHING:
        raw_points = get_subpixel_path_points(chain, cell_w, cell_h, gray)
        smooth_points = smooth_path_with_spline(raw_points, 
                                               num_points=len(raw_points) * 3,
                                               smoothing_factor=SMOOTHING_FACTOR)
        result = draw_smooth_path(result, smooth_points, 
                                  color=(0, 0, 255),
                                  width=PATH_LINE_WIDTH,
                                  show_nodes=True,
                                  node_radius=PATH_POINT_RADIUS,
                                  show_gradient=True)
    else:
        points = []
        for (r, c) in chain:
            cx = c * cell_w + cell_w // 2
            cy = r * cell_h + cell_h // 2
            points.append((cx, cy))
        result = draw_smooth_path(result, points,
                                  color=(0, 0, 255),
                                  width=PATH_LINE_WIDTH,
                                  show_nodes=True,
                                  node_radius=PATH_POINT_RADIUS,
                                  show_gradient=True)

    # 7. 绘制信息文字
    info_lines = get_info_lines(stats, sample_rect)
    
    if info_lines:
        text_config = {
            'scale': TEXT_SCALE,
            'thickness': TEXT_THICKNESS,
            'line_spacing': TEXT_LINE_SPACING,
            'start_y': TEXT_START_Y,
            'color': TEXT_COLOR,
            'bg_color': TEXT_BG_COLOR,
            'bg_alpha': TEXT_BG_ALPHA
        }
        draw_info_text(result, info_lines, text_config)

    print("\n" + "=" * 60)
    print("📊 最终统计")
    print("=" * 60)
    print(f"  样本均值: {reference_mean:.2f}")
    print(f"  样本标准差: {reference_std:.2f}")
    print(f"  相似度阈值: {threshold:.2f}")
    print(f"  网格尺寸: {cell_w} x {cell_h}")
    print(f"  网格数量: {grid_w} x {grid_h}")
    print(f"  有效起点: {len(start_keys)}")
    print(f"  访问节点: {len(all_visited)}")
    print(f"  路径总数: {len(all_paths)}")
    print(f"  最优路径长度: {len(chain)}")
    print(f"  路径评分: {best_details['total']:.3f}")
    print("=" * 60 + "\n")

    return result, chain, stats

# ======================== 交互选择器 ========================

class InteractiveSelector:
    """处理缩放浏览 + 样本选择的完整交互状态机"""
    
    def __init__(self, image_path):
        if not os.path.isfile(image_path):
            print(f"[错误] 找不到图片: {image_path}")
            sys.exit(1)
        self.orig = cv2.imread(image_path)
        if self.orig is None:
            print(f"[错误] 无法读取图片: {image_path}")
            sys.exit(1)

        self.small, self.full, self.init_scale = fit_size(self.orig)
        self.view = self.small.copy()
        self.zoom = None
        self.scale = self.init_scale

        self.mode = None
        self.pt_start = None
        self.pt_end = None
        self.drawing = False
        self.sample_rect = None
        self.done = False

    def on_mouse(self, event, x, y, flags, _):
        if self.done:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.mode = 'select'
            self.pt_start = (x, y)
            self.pt_end = (x, y)
            self._redraw_with_rect()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.mode = 'zoom'
            self.pt_start = (x, y)
            self.pt_end = (x, y)
            self._redraw_with_rect()
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.pt_end = (x, y)
            self._redraw_with_rect()
        elif event == cv2.EVENT_LBUTTONUP and self.mode == 'select':
            self.drawing = False
            self.pt_end = (x, y)
            self._finalize_select()
        elif event == cv2.EVENT_RBUTTONUP and self.mode == 'zoom':
            self.drawing = False
            self.pt_end = (x, y)
            self._finalize_zoom()

    def _redraw_with_rect(self):
        base = self.small.copy() if self.zoom is None else self._crop_zoom()
        self.view = base.copy()

        if self.pt_start is None or self.pt_end is None:
            cv2.imshow("RegionGrowing", self.view)
            return

        x1, y1 = self.pt_start
        x2, y2 = self.pt_end
        color = (0, 255, 0) if self.mode == 'select' else (255, 165, 0)
        cv2.rectangle(
            self.view,
            (min(x1, x2), min(y1, y2)),
            (max(x1, x2), max(y1, y2)),
            color,
            2,
        )
        cv2.imshow("RegionGrowing", self.view)

    def _crop_zoom(self):
        zx1, zy1, zx2, zy2 = self.zoom
        crop = self.full[zy1:zy2, zx1:zx2]
        vh, vw = self.small.shape[:2]
        resized = cv2.resize(crop, (vw, vh), interpolation=cv2.INTER_LINEAR)
        return resized

    def _finalize_zoom(self):
        sx = int(min(self.pt_start[0], self.pt_end[0]) / self.scale)
        sy = int(min(self.pt_start[1], self.pt_end[1]) / self.scale)
        ex = int(max(self.pt_start[0], self.pt_end[0]) / self.scale)
        ey = int(max(self.pt_start[1], self.pt_end[1]) / self.scale)
        if (ex - sx) > 10 and (ey - sy) > 10:
            self.zoom = (sx, sy, ex, ey)
            self.scale = self.small.shape[1] / (ex - sx)
            self.view = self._crop_zoom()
            print(f"[放大] 区域 ({sx},{sy})-({ex},{ey})，缩放 ×{self.scale:.2f}")
        else:
            self.view = self.small.copy()
        cv2.imshow("RegionGrowing", self.view)

    def _finalize_select(self):
        if self.zoom is None:
            sx = int(min(self.pt_start[0], self.pt_end[0]) / self.scale)
            sy = int(min(self.pt_start[1], self.pt_end[1]) / self.scale)
            ex = int(max(self.pt_start[0], self.pt_end[0]) / self.scale)
            ey = int(max(self.pt_start[1], self.pt_end[1]) / self.scale)
        else:
            zx1, zy1, zx2, zy2 = self.zoom
            zw, zh = zx2 - zx1, zy2 - zy1
            vh, vw = self.small.shape[:2]
            rx1 = int(min(self.pt_start[0], self.pt_end[0]) / vw * zw + zx1)
            ry1 = int(min(self.pt_start[1], self.pt_end[1]) / vh * zh + zy1)
            rx2 = int(max(self.pt_start[0], self.pt_end[0]) / vw * zw + zx1)
            ry2 = int(max(self.pt_start[1], self.pt_end[1]) / vh * zh + zy1)
            sx, sy, ex, ey = rx1, ry1, rx2, ry2

        if (ex - sx) > 5 and (ey - sy) > 5:
            self.sample_rect = (sx, sy, ex, ey)
            self.done = True
            print(f"[样本] 原图区域 ({sx},{sy})-({ex},{ey})")
        else:
            print("[提示] 选区太小，请重新框选")
            self.view = self.small.copy() if self.zoom is None else self._crop_zoom()
            cv2.imshow("RegionGrowing", self.view)

    def run(self):
        cv2.namedWindow("RegionGrowing", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("RegionGrowing", self.on_mouse)
        cv2.imshow("RegionGrowing", self.view)

        print("=" * 50)
        print("  右键拖框 = 放大     左键拖框 = 选择样本区域")
        print("  选完样本后自动进入处理流程")
        print("=" * 50)

        while not self.done:
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                sys.exit(0)
            if key == ord('r') or key == ord('R'):
                self.zoom = None
                self.scale = self.init_scale
                self.view = self.small.copy()
                cv2.imshow("RegionGrowing", self.view)

        cv2.destroyAllWindows()
        return self.sample_rect

# ======================== 结果窗口交互 ========================

class ResultViewer:
    """结果窗口交互：右键放大（带矩形框显示），左键复原，空格重来"""
    
    def __init__(self, result_image, original_image, window_name="Result", reset_callback=None):
        self.orig_result = result_image.copy()
        self.orig_full = original_image.copy()
        self.window_name = window_name
        self.reset_callback = reset_callback
        
        # 当前显示的缩放状态
        self.zoom = None          # (x1, y1, x2, y2) 原图坐标
        self.scale = 1.0
        self.view_offset_x = 0    # 当前视图在放大图像中的偏移x
        self.view_offset_y = 0    # 当前视图在放大图像中的偏移y
        self.view = self.orig_result.copy()
        self.drawing = False
        self.pt_start = None
        self.pt_end = None
        self.is_panning = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.zoom_history = []    # 缩放历史，支持回退
        
        # 创建窗口并设置鼠标回调
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        cv2.imshow(self.window_name, self.view)
        cv2.setWindowTitle(self.window_name, "Result (左键拖框放大 | 中键拖拽移动 | 双击复原 | 空格重来)")
    
    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_MBUTTONDOWN:
            self.is_panning = True
            self.last_mouse_x = x
            self.last_mouse_y = y
        elif event == cv2.EVENT_MBUTTONUP:
            self.is_panning = False
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self._reset_view()
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt_start = (x, y)
            self.pt_end = (x, y)
            self._redraw_with_rect()
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.pt_end = (x, y)
            self._finalize_zoom()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning:
                if self.zoom is None:
                    return
                dx = x - self.last_mouse_x
                dy = y - self.last_mouse_y
                self.view_offset_x += dx
                self.view_offset_y += dy
                self.last_mouse_x = x
                self.last_mouse_y = y
                self._update_panned_view()
            elif self.drawing:
                self.pt_end = (x, y)
                self._redraw_with_rect()
    
    def _reset_view(self):
        """恢复原始视图"""
        self.zoom = None
        self.scale = 1.0
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.zoom_history = []
        self.view = self.orig_result.copy()
        cv2.imshow(self.window_name, self.view)
        cv2.setWindowTitle(self.window_name, "Result (左键拖框放大 | 中键拖拽移动 | 双击复原 | 空格重来)")
        print("[结果] 已恢复原始视图")
    
    def _redraw_with_rect(self):
        """
        拖动时显示矩形框（关键改进：矩形框实时可见）
        """
        # 获取当前基础视图
        if self.zoom is None:
            base = self.orig_result.copy()
        else:
            zx1, zy1, zx2, zy2 = self.zoom
            crop = self.orig_result[zy1:zy2, zx1:zx2]
            h, w = self.orig_result.shape[:2]
            base = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        
        self.view = base.copy()
        
        # 绘制矩形框（如果正在拖拽）
        if self.drawing and self.pt_start is not None and self.pt_end is not None:
            x1, y1 = self.pt_start
            x2, y2 = self.pt_end
            
            # 计算矩形框尺寸
            rect_w = abs(x2 - x1)
            rect_h = abs(y2 - y1)
            
            # 如果矩形太小，显示提示信息
            if rect_w < 10 or rect_h < 10:
                # 绘制小矩形框（橙色）
                cv2.rectangle(
                    self.view,
                    (min(x1, x2), min(y1, y2)),
                    (max(x1, x2), max(y1, y2)),
                    (0, 165, 255),  # 橙色
                    2,
                )
                # 显示提示文字
                cv2.putText(
                    self.view,
                    "框选区域太小，请继续拖动",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )
            else:
                # 绘制矩形框（亮橙色，加粗）
                cv2.rectangle(
                    self.view,
                    (min(x1, x2), min(y1, y2)),
                    (max(x1, x2), max(y1, y2)),
                    (0, 165, 255),  # 亮橙色
                    3,  # 加粗边框
                )
                
                # 绘制半透明填充（让框选区域更明显）
                overlay = self.view.copy()
                cv2.rectangle(
                    overlay,
                    (min(x1, x2), min(y1, y2)),
                    (max(x1, x2), max(y1, y2)),
                    (0, 165, 255),
                    -1  # 填充
                )
                cv2.addWeighted(overlay, 0.15, self.view, 0.85, 0, self.view)
                
                # 显示框选尺寸信息
                size_text = f"{rect_w} x {rect_h}"
                cv2.putText(
                    self.view,
                    size_text,
                    (min(x1, x2) + 5, min(y1, y2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),  # 黄色
                    2
                )
                
                # 显示操作提示
                cv2.putText(
                    self.view,
                    "释放右键放大此区域",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
        
        cv2.imshow(self.window_name, self.view)
    
    def _finalize_zoom(self):
        """执行放大"""
        if self.pt_start is None or self.pt_end is None:
            return
        
        # 获取当前视图的尺寸
        h, w = self.view.shape[:2]
        
        # 计算框选区域在视图中的比例
        x1 = min(self.pt_start[0], self.pt_end[0])
        y1 = min(self.pt_start[1], self.pt_end[1])
        x2 = max(self.pt_start[0], self.pt_end[0])
        y2 = max(self.pt_start[1], self.pt_end[1])
        
        # 检查框选是否足够大
        if (x2 - x1) < 15 or (y2 - y1) < 15:
            print("[提示] 框选区域太小，请重新选择（至少15x15像素）")
            cv2.imshow(self.window_name, self.view)
            return
        
        # 计算在原图中的对应区域
        if self.zoom is None:
            # 当前显示的是原始结果图
            scale_x = self.orig_result.shape[1] / w
            scale_y = self.orig_result.shape[0] / h
            sx = int(x1 * scale_x)
            sy = int(y1 * scale_y)
            ex = int(x2 * scale_x)
            ey = int(y2 * scale_y)
        else:
            # 当前显示的是放大后的裁剪图
            zx1, zy1, zx2, zy2 = self.zoom
            zw = zx2 - zx1
            zh = zy2 - zy1
            sx = int(x1 / w * zw + zx1)
            sy = int(y1 / h * zh + zy1)
            ex = int(x2 / w * zw + zx1)
            ey = int(y2 / h * zh + zy1)
        
        # 确保坐标在图像范围内
        sx = max(0, min(sx, self.orig_result.shape[1] - 1))
        sy = max(0, min(sy, self.orig_result.shape[0] - 1))
        ex = max(0, min(ex, self.orig_result.shape[1] - 1))
        ey = max(0, min(ey, self.orig_result.shape[0] - 1))
        
        # 确保宽高至少为20像素
        if (ex - sx) < 20 or (ey - sy) < 20:
            print("[提示] 放大区域太小，请选择更大的区域")
            cv2.imshow(self.window_name, self.view)
            return
        
        # 保存缩放历史
        self.zoom_history.append(self.zoom)
        
        # 执行放大
        self.zoom = (sx, sy, ex, ey)
        crop = self.orig_result[sy:ey, sx:ex]
        
        # 缩放到窗口大小
        disp_h, disp_w = self.orig_result.shape[:2]
        self.view = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        
        cv2.imshow(self.window_name, self.view)
        cv2.setWindowTitle(self.window_name, 
            f"Result (放大 {ex-sx}x{ey-sy} | 左键复原 | 右键继续放大 | 空格重来)")
        print(f"[结果放大] 区域 ({sx},{sy})-({ex},{ey})，尺寸 {ex-sx}x{ey-sy}")
        
        # 清除拖拽状态
        self.pt_start = None
        self.pt_end = None
    
    def _update_panned_view(self):
        if self.zoom is None:
            self.view = self.orig_result.copy()
            cv2.imshow(self.window_name, self.view)
            return

        zx1, zy1, zx2, zy2 = self.zoom
        crop_w = zx2 - zx1
        crop_h = zy2 - zy1
        disp_h, disp_w = self.orig_result.shape[:2]

        if crop_w <= disp_w:
            min_offset_x = -(disp_w - crop_w)
            max_offset_x = 0
        else:
            min_offset_x = 0
            max_offset_x = crop_w - disp_w

        if crop_h <= disp_h:
            min_offset_y = -(disp_h - crop_h)
            max_offset_y = 0
        else:
            min_offset_y = 0
            max_offset_y = crop_h - disp_h

        self.view_offset_x = max(min_offset_x, min(self.view_offset_x, max_offset_x))
        self.view_offset_y = max(min_offset_y, min(self.view_offset_y, max_offset_y))

        canvas = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        sx = zx1 + int(self.view_offset_x)
        sy = zy1 + int(self.view_offset_y)
        ex = sx + disp_w
        ey = sy + disp_h

        crop_sx = max(0, sx - zx1)
        crop_sy = max(0, sy - zy1)
        crop_ex = min(crop_w, ex - zx1)
        crop_ey = min(crop_h, ey - zy1)

        if crop_ex > crop_sx and crop_ey > crop_sy:
            crop = self.orig_result[zy1 + crop_sy:zy1 + crop_ey, zx1 + crop_sx:zx1 + crop_ex]
            canvas_sx = max(0, -int(self.view_offset_x))
            canvas_sy = max(0, -int(self.view_offset_y))
            canvas[canvas_sy:canvas_sy + (crop_ey - crop_sy),
                   canvas_sx:canvas_sx + (crop_ex - crop_sx)] = crop

        self.view = canvas
        cv2.imshow(self.window_name, self.view)
    
    def run(self):
        """运行结果查看器"""
        cv2.setWindowTitle(self.window_name, "Result (左键复原 | 右键放大 | 空格重来)")
        
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC退出
                return 'exit'
            elif key == 32:  # 空格键重来
                print("[重来] 按空格键重新选择样本...")
                if self.reset_callback is not None:
                    self.reset_callback()
                return 'reset'
            elif key == ord('r') or key == ord('R'):
                # R键也可以复原
                self._reset_view()
            elif key == ord('z') or key == ord('Z'):
                # Z键回退上一级缩放
                if self.zoom_history:
                    prev_zoom = self.zoom_history.pop()
                    self.zoom = prev_zoom
                    if self.zoom is not None:
                        zx1, zy1, zx2, zy2 = self.zoom
                        crop = self.orig_result[zy1:zy2, zx1:zx2]
                        disp_h, disp_w = self.orig_result.shape[:2]
                        self.view = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                        cv2.imshow(self.window_name, self.view)
                        cv2.setWindowTitle(self.window_name, 
                            f"Result (回退 | 左键复原 | 右键放大 | 空格重来)")
                        print(f"[结果] 回退到上一级缩放")
                else:
                    print("[提示] 没有更早的缩放历史")
        
        return 'continue'

# ======================== 滑块回调 ========================

def on_slider_change(x):
    global slider_filter, slider_factor, current_image, current_sample_rect
    
    slider_filter = cv2.getTrackbarPos('Filter Threshold', 'Result')
    slider_factor = cv2.getTrackbarPos('Growth Factor', 'Result') / 10.0
    
    if current_image is not None and current_sample_rect is not None:
        update_result()

def update_result():
    global current_result, current_chain, current_image, current_sample_rect, result_viewer
    
    result, chain, stats = region_growing_with_params(
        current_image, 
        current_sample_rect, 
        slider_filter, 
        slider_factor,
        show_preview=False
    )
    
    if stats is not None:
        current_result = result
        current_chain = chain
        
        # 更新结果查看器
        if 'result_viewer' in globals() and result_viewer is not None:
            result_viewer.orig_result = result.copy()
            result_viewer._reset_view()
        else:
            result_small, _, _ = fit_size(result)
            cv2.imshow("Result", result_small)
        
        cv2.setWindowTitle("Result", 
            f"Result | Filter={slider_filter} Factor={slider_factor:.1f} Chain={len(chain)}")

# ======================== 主函数 ========================

def main():
    global current_image, current_sample_rect, current_result, current_chain, result_viewer
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    print(f"[启动] 加载图片: {image_path}")

    print("\n" + "=" * 50)
    print("📝 当前配置:")
    print(f"  - 邻域类型: {'8邻域' if USE_8_NEIGHBOR else '4邻域'}")
    print(f"  - 路径平滑: {'启用' if ENABLE_PATH_SMOOTHING else '禁用'}")
    print(f"  - 平滑因子: {SMOOTHING_FACTOR}")
    print(f"  - 显示信息: {', '.join(DISPLAY_INFO) if DISPLAY_INFO else '无'}")
    print("  (修改文件开头的配置参数可调整)")
    print("=" * 50 + "\n")

    # 重置函数（供空格键调用）
    def reset_and_restart():
        """重置程序，重新选择样本"""
        cv2.destroyAllWindows()
        main()
    
    # 第一次选择样本
    selector = InteractiveSelector(image_path)
    sample_rect = selector.run()

    if sample_rect is None:
        print("[退出] 未选择样本区域")
        return

    current_image = selector.full
    current_sample_rect = sample_rect
    
    print("\n[处理中] 执行智能路径追踪算法...")
    result, chain, stats = region_growing_with_params(
        current_image, 
        sample_rect, 
        slider_filter, 
        slider_factor,
        show_preview=True
    )
    
    if stats is None:
        print("[错误] 处理失败")
        return
    
    current_result = result
    current_chain = chain

    # ---- 创建结果窗口（带交互功能） ----
    result_small, _, _ = fit_size(result)
    
    # 创建结果查看器，传入重置回调
    result_viewer = ResultViewer(result, result_small, "Result", reset_callback=reset_and_restart)
    
    # 添加滑块
    cv2.createTrackbar('Filter Threshold', 'Result', slider_filter, 255, on_slider_change)
    cv2.createTrackbar('Growth Factor', 'Result', int(slider_factor * 10), 50, on_slider_change)
    
    print("\n" + "=" * 50)
    print("🎛️ 滑块控制:")
    print("  - Filter Threshold: 过滤 > 该值的像素 (0-255)")
    print("  - Growth Factor: 生长阈值因子 (0-5.0，实际值 = 滑块值/10)")
    print("  调整滑块 → 实时更新结果")
    print("=" * 50)
    print("📌 结果窗口操作:")
    print("  - 左键单击: 恢复原始视图")
    print("  - 右键拖框: 放大查看细节（矩形框实时显示）")
    print("  - 空格键: 重新选择样本 (重来)")
    print("  - R键: 恢复原始视图")
    print("  - Z键: 回退上一级缩放")
    print("=" * 50)
    print("📌 图例:")
    print("  - 淡黄色区域: 路径经过的格子")
    print("  - 彩色渐变线: 平滑后的路径（青→红渐变）")
    print("  - S: 起点, E: 终点")
    print("  - 绿色框: 样本区域")
    print("=" * 50)
    print("按 ESC 退出，按 S 保存截图，按 空格键 重新选择样本")

    # ---- 主循环 ----
    while True:
        key = cv2.waitKey(100) & 0xFF
        
        if key == 27:  # ESC退出
            break
        elif key == 32:  # 空格键重来
            print("[重来] 重新选择样本区域...")
            cv2.destroyAllWindows()
            main()
            return
        elif key == ord('s') or key == ord('S'):
            if current_result is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                base, ext = os.path.splitext(image_path)
                snap = f"{base}_snap_{ts}{ext}"
                cv2.imwrite(snap, current_result)
                print(f"[截图] 已保存: {snap}")
        elif key == ord('r') or key == ord('R'):
            # 重置视图（调用ResultViewer的复位方法）
            if 'result_viewer' in globals() and result_viewer is not None:
                result_viewer._reset_view()

    cv2.destroyAllWindows()
    print("[退出] 再见！")

if __name__ == "__main__":
    main()