"""
可以用main.py测试FindPath模块的GUI工具
FindPath.py - 路径检测核心类
"""

import cv2
import numpy as np
import math
import heapq
import time
from collections import deque
import skimage.morphology as morph


class FindPath:
    """
    路径检测核心类
    """
    
    def __init__(self, image, threshold=180, window_h=20, window_w=25, 
                 prune_threshold=6, keep_paths=5, invert=False, verbose=False):
        """
        初始化路径检测器
        
        参数:
            image: 图片路径(str) 或 numpy数组 (BGR或灰度)
            threshold: 灰度阈值 (默认180)
            window_h: 扫描窗口高度 (默认20)
            window_w: 扫描窗口宽度 (默认25)
            prune_threshold: 去毛刺阈值 (默认6)
            keep_paths: 保留路径数量 (默认5)
            invert: 是否反转颜色 (默认False)
            verbose: 是否显示进度信息 (默认False)
        """
        self.threshold = threshold
        self.window_h = window_h
        self.window_w = window_w
        self.prune_threshold = prune_threshold
        self.keep_paths = keep_paths
        self.invert = invert
        self.verbose = verbose
        
        # 加载图像
        if isinstance(image, str):
            self.img_color = cv2.imread(image, cv2.IMREAD_COLOR)
            if self.img_color is None:
                raise FileNotFoundError(f"无法读取图片: {image}")
        else:
            if len(image.shape) == 2:
                self.img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                self.img_color = image.copy()
        
        self.img_gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        self.img_h, self.img_w = self.img_gray.shape
        
        # 结果存储
        self.binary_image = None
        self.path_image = None
        self.paths = []
        self.path_count = 0
        self.path_infos = []
        self.skeleton = None
        self._processed = False
        
        # 8邻域邻居（带权重）
        self.NEIGHBORS = [
            (-1, -1, 1.41421356),
            (-1, 0, 1.0),
            (-1, 1, 1.41421356),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (1, -1, 1.41421356),
            (1, 0, 1.0),
            (1, 1, 1.41421356),
        ]
    
    def process(self):
        """执行完整的路径检测流程"""
        start_time = time.time()
        
        if self.verbose:
            print("步骤1: 扫描生成二值图...")
        
        # 步骤1: 区域扫描生成二值图
        self.binary_image = self._region_scan()
        
        if self.verbose:
            print(f"  二值图生成完成, 像素: {np.sum(self.binary_image == 255)}")
            print("步骤2: 提取路径...")
        
        # 步骤2: 提取路径
        self._extract_paths()
        
        self._processed = True
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"✅ 处理完成! 找到 {self.path_count} 条路径, 耗时: {self._format_time(elapsed)}")
        
        return self
    
    # ==================== 步骤1: 区域扫描 ====================
    
    def _region_scan(self):
        """区域扫描算法"""
        h, w = self.img_h, self.img_w
        binary_img = np.zeros((h, w), dtype=np.uint8)
        
        rect_h = self.window_h
        rect_w = self.window_w
        
        if rect_w < 1 or rect_h < 1:
            rect_w = 20
            rect_h = 20
        
        # 使用积分图加速
        integral = np.zeros((h + 1, w + 1), dtype=np.float64)
        integral[1:, 1:] = np.cumsum(np.cumsum(self.img_gray.astype(np.float64), axis=0), axis=1)
        
        total_rows = h - rect_h + 1
        processed = 0
        
        for y in range(h - rect_h + 1):
            for x in range(w - rect_w + 1):
                sum_val = (integral[y + rect_h, x + rect_w] - 
                           integral[y, x + rect_w] - 
                           integral[y + rect_h, x] + 
                           integral[y, x])
                area = rect_h * rect_w
                avg_val = sum_val / area
                
                if avg_val <= self.threshold:
                    binary_img[y:y + rect_h, x:x + rect_w] = 255
            
            processed += 1
            if self.verbose and processed % max(1, total_rows // 20) == 0:
                progress = int((processed / total_rows) * 100)
                print(f"  扫描进度: {progress}%")
        
        # 如果启用反转
        if self.invert:
            binary_img = cv2.bitwise_not(binary_img)
        
        return binary_img
    
    # ==================== 步骤2: 路径提取 ====================
    
    def _extract_paths(self):
        """提取路径"""
        
        # 1. 骨骼化
        binary_01 = (self.binary_image == 255).astype(np.uint8)
        skeleton_01 = morph.skeletonize(binary_01)
        skeleton_255 = (skeleton_01 * 255).astype(np.uint8)
        
        # 2. 去毛刺
        skeleton_pruned = self._prune_short_branches(skeleton_255, self.prune_threshold)
        self.skeleton = skeleton_pruned
        
        # 3. 提取所有连通路径
        n_labels, labels = cv2.connectedComponents(skeleton_pruned.astype(np.uint8), connectivity=8)
        
        # 4. 对每条路径使用Dijkstra提取最长路径
        all_main_paths = []
        path_infos = []
        
        for comp_id in range(1, n_labels):
            comp_mask = labels == comp_id
            comp_points = np.argwhere(comp_mask)
            if comp_points.shape[0] == 0:
                continue
            
            # 使用Dijkstra算法找最长路径
            main_path, (start_pt, end_pt) = self._longest_path_dijkstra(comp_mask)
            
            # 转换路径点格式: (y, x) -> (x, y) 用于绘制
            converted_path = []
            for pt in main_path:
                y, x = pt
                if 0 <= x < self.img_w and 0 <= y < self.img_h:
                    converted_path.append((x, y))
            
            if len(converted_path) >= 10:
                all_main_paths.append(converted_path)
                path_infos.append({
                    'id': len(path_infos) + 1,
                    'pixels': int(comp_points.shape[0]),
                    'path_len': len(converted_path),
                    'start': (start_pt[1], start_pt[0]) if start_pt else None,
                    'end': (end_pt[1], end_pt[0]) if end_pt else None
                })
        
        # 5. 去分叉
        merged_paths = self._remove_branches(all_main_paths)
        
        # 6. 按长度排序，保留前N条
        if len(merged_paths) > 0:
            merged_paths.sort(key=len, reverse=True)
            self.paths = merged_paths[:self.keep_paths]
        else:
            self.paths = []
        
        self.path_infos = path_infos
        self.path_count = len(self.paths)
        
        # 7. 生成路径标记图
        self.path_image = self._draw_paths()
    
    def _draw_paths(self):
        """绘制路径标记图"""
        img = self.img_color.copy()
        colors = self._generate_colors(len(self.paths))
        
        min_width = self.window_h
        line_thickness = max(1, min_width // 2)
        circle_radius = max(2, min_width // 3)
        circle_thickness = max(1, min_width // 5)
        font_scale = max(0.3, min_width / 30)
        font_thickness = max(1, min_width // 5)
        
        for idx, path in enumerate(self.paths):
            if len(path) < 2:
                continue
            
            color = colors[idx % len(colors)]
            
            valid_path = []
            for pt in path:
                x, y = pt
                if 0 <= x < self.img_w and 0 <= y < self.img_h:
                    valid_path.append((x, y))
            
            if len(valid_path) < 2:
                continue
            
            for i in range(len(valid_path) - 1):
                cv2.line(img, valid_path[i], valid_path[i+1], color, line_thickness)
            
            # 起点：黄色空心圆
            cv2.circle(img, valid_path[0], circle_radius, (0, 255, 255), circle_thickness)
            
            # 终点：黄色实心圆
            cv2.circle(img, valid_path[-1], circle_radius, (0, 255, 255), -1)
            
            # 路径编号
            mid_idx = len(valid_path) // 2
            cv2.putText(img, f"#{idx+1}", (valid_path[mid_idx][0] + 5, valid_path[mid_idx][1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        
        return img
    
    # ==================== 辅助函数 ====================
    
    def _iter_neighbors(self, mask, y, x):
        h, w = mask.shape
        for dy, dx, weight in self.NEIGHBORS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                yield ny, nx, weight
    
    def _find_endpoints(self, skeleton_bool):
        skel_float = (skeleton_bool.astype(np.float32)) * 255.0
        kernel = np.array(
            [[1, 1, 1],
             [1, 0, 1],
             [1, 1, 1]],
            dtype=np.float32,
        )
        neighbor_count = cv2.filter2D(skel_float, -1, kernel)
        endpoints_mask = (np.abs(neighbor_count - 1.0) < 0.5) & (skel_float == 255.0)
        return np.argwhere(endpoints_mask)
    
    def _trace_branch_from_endpoint(self, skeleton_bool, endpoint):
        h, w = skeleton_bool.shape
        cy, cx = int(endpoint[0]), int(endpoint[1])
        prev = None
        branch = []
        
        while True:
            branch.append((cy, cx))
            neighbors = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and skeleton_bool[ny, nx]:
                        if prev is None or (ny, nx) != prev:
                            neighbors.append((ny, nx))
            
            if len(neighbors) != 1:
                break
            
            prev = (cy, cx)
            cy, cx = neighbors[0]
        
        return branch, len(branch)
    
    def _prune_short_branches(self, skeleton_255, max_branch_length):
        skel = skeleton_255.copy().astype(bool)
        
        while True:
            endpoints = self._find_endpoints(skel)
            if len(endpoints) == 0:
                break
            
            removed = False
            candidates = []
            for ep in endpoints:
                branch, length = self._trace_branch_from_endpoint(skel, ep)
                candidates.append((length, branch))
            
            candidates.sort(key=lambda item: item[0])
            for length, branch in candidates:
                if length <= max_branch_length:
                    for y, x in branch:
                        skel[y, x] = False
                    removed = True
                    break
            
            if not removed:
                break
        
        return skel.astype(np.uint8) * 255
    
    def _dijkstra_farthest(self, mask, start, need_prev=False):
        pq = [(0.0, start)]
        dist = {start: 0.0}
        prev = {} if need_prev else None
        farthest = start
        
        while pq:
            cur_d, cur = heapq.heappop(pq)
            if cur_d != dist.get(cur, np.inf):
                continue
            if cur_d > dist[farthest]:
                farthest = cur
            
            cy, cx = cur
            for ny, nx, w in self._iter_neighbors(mask, cy, cx):
                nxt = (ny, nx)
                nd = cur_d + w
                if nd < dist.get(nxt, np.inf):
                    dist[nxt] = nd
                    if need_prev:
                        prev[nxt] = cur
                    heapq.heappush(pq, (nd, nxt))
        
        return farthest, dist, prev
    
    def _longest_path_dijkstra(self, mask):
        points = np.argwhere(mask)
        if points.shape[0] == 0:
            return [], (None, None)
        if points.shape[0] == 1:
            p = (int(points[0, 0]), int(points[0, 1]))
            return [p], (p, p)
        
        seed = (int(points[0, 0]), int(points[0, 1]))
        a, _, _ = self._dijkstra_farthest(mask, seed, need_prev=False)
        b, _, prev = self._dijkstra_farthest(mask, a, need_prev=True)
        
        path = [b]
        cur = b
        while cur != a and cur in prev:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path, (a, b)
    
    def _remove_branches(self, all_paths):
        if len(all_paths) <= 1:
            return all_paths
        
        merged = []
        used = [False] * len(all_paths)
        
        for i in range(len(all_paths)):
            if used[i]:
                continue
            
            connected = [i]
            used[i] = True
            
            changed = True
            while changed:
                changed = False
                for j in range(len(all_paths)):
                    if used[j]:
                        continue
                    
                    for idx in connected:
                        if self._is_connected(all_paths[idx], all_paths[j], 5):
                            connected.append(j)
                            used[j] = True
                            changed = True
                            break
            
            if connected:
                best_path = max([all_paths[idx] for idx in connected], key=len)
                merged.append(best_path)
        
        return merged
    
    def _is_connected(self, path1, path2, max_dist):
        if len(path1) == 0 or len(path2) == 0:
            return False
        
        endpoints1 = [path1[0], path1[-1]]
        endpoints2 = [path2[0], path2[-1]]
        
        for p1 in endpoints1:
            for p2 in endpoints2:
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist <= max_dist:
                    return True
        return False
    
    def _generate_colors(self, count):
        colors = [
            (0, 0, 255),
            (0, 165, 255),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (128, 0, 255),
            (0, 200, 0),
            (255, 128, 0),
        ]
        
        result = []
        for i in range(count):
            result.append(colors[i % len(colors)])
        return result
    
    def _format_time(self, seconds):
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
    
    # ==================== 输出接口 ====================
    
    def get_binary_image(self):
        if not self._processed:
            self.process()
        return self.binary_image
    
    def get_path_image(self):
        if not self._processed:
            self.process()
        return self.path_image
    
    def get_paths(self):
        if not self._processed:
            self.process()
        return self.paths
    
    def get_path_count(self):
        if not self._processed:
            self.process()
        return self.path_count
    
    def get_path_infos(self):
        if not self._processed:
            self.process()
        return self.path_infos
    
    def get_skeleton(self):
        if not self._processed:
            self.process()
        return self.skeleton
    
    def save_results(self, binary_path="binary.png", path_path="paths.png"):
        if not self._processed:
            self.process()
        
        if self.binary_image is not None:
            cv2.imwrite(binary_path, self.binary_image)
        if self.path_image is not None:
            cv2.imwrite(path_path, self.path_image)
        
        return binary_path, path_path


# ==================== 便捷函数 ====================

def find_paths(image, threshold=180, window_h=20, window_w=25, 
               prune_threshold=6, keep_paths=5, invert=False, verbose=False):
    """
    快速检测路径的便捷函数
    """
    finder = FindPath(image, threshold, window_h, window_w, 
                      prune_threshold, keep_paths, invert, verbose)
    finder.process()
    return finder.binary_image, finder.path_image, finder.paths, finder.path_count


# ==================== 测试入口 ====================

if __name__ == "__main__":
    print("FindPath 类已加载")
    print("使用示例:")
    print("  finder = FindPath('image.jpg', threshold=180, window_h=20, window_w=25)")
    print("  finder.process()")
    print("  binary_img = finder.binary_image")
    print("  path_img = finder.path_image")
    print("  count = finder.path_count")