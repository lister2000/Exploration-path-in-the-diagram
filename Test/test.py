import ctypes
import heapq
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

import cv2
import numpy as np
from skimage.morphology import skeletonize


# ==================== 原有的处理函数 ====================
def to_binary(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float((binary > 0).mean()) > 0.5:
        binary = 255 - binary
    return binary


def find_endpoints(skeleton_bool):
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


NEIGHBORS = [
    (-1, -1, 1.41421356),
    (-1, 0, 1.0),
    (-1, 1, 1.41421356),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, 1.41421356),
    (1, 0, 1.0),
    (1, 1, 1.41421356),
]


def iter_neighbors(mask, y, x):
    h, w = mask.shape
    for dy, dx, weight in NEIGHBORS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
            yield ny, nx, weight


def neighbor_degree(mask, y, x):
    return sum(1 for _ in iter_neighbors(mask, y, x))


def choose_closed_anchor(mask, main_path, junctions):
    junction_set = set(junctions)

    loop_points = []
    for y, x in main_path:
        if (y, x) in junction_set:
            continue
        if neighbor_degree(mask, int(y), int(x)) == 2:
            loop_points.append((int(y), int(x)))
    if loop_points:
        return loop_points[len(loop_points) // 2]

    points = np.argwhere(mask)
    for y, x in points:
        p = (int(y), int(x))
        if p not in junction_set:
            return p

    if points.shape[0] > 0:
        return int(points[0, 0]), int(points[0, 1])
    return None


def peel_to_core(mask):
    core = mask.copy()
    while True:
        ys, xs = np.where(core)
        remove = []
        for y, x in zip(ys, xs):
            if neighbor_degree(core, int(y), int(x)) <= 1:
                remove.append((int(y), int(x)))
        if not remove:
            break
        for y, x in remove:
            core[y, x] = False
    return core


def core_path_points(mask):
    core = peel_to_core(mask)
    return [(int(y), int(x)) for y, x in np.argwhere(core)], core


def path_color_pair(index):
    color_pairs = [
        ((0, 255, 0), (0, 0, 255)),
        ((0, 220, 255), (255, 64, 0)),
        ((255, 255, 0), (255, 0, 180)),
        ((255, 128, 0), (0, 255, 255)),
        ((255, 0, 255), (0, 200, 0)),
        ((0, 255, 255), (255, 0, 0)),
        ((180, 255, 60), (180, 0, 255)),
        ((120, 255, 120), (255, 60, 60)),
    ]
    main_color, branch_color = color_pairs[index % len(color_pairs)]
    return main_color, branch_color


def largest_component(mask):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return mask.copy(), labels
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    return labels == best, labels


def trace_branch_from_endpoint(skeleton_bool, endpoint):
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


def prune_short_branches(skeleton_bool, max_branch_length):
    skel = skeleton_bool.copy()

    while True:
        endpoints = find_endpoints(skel)
        if len(endpoints) == 0:
            break

        removed = False
        candidates = []
        for ep in endpoints:
            branch, length = trace_branch_from_endpoint(skel, ep)
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

    return skel


def dijkstra_farthest(mask, start, need_prev=False):
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
        for ny, nx, w in iter_neighbors(mask, cy, cx):
            nxt = (ny, nx)
            nd = cur_d + w
            if nd < dist.get(nxt, np.inf):
                dist[nxt] = nd
                if need_prev:
                    prev[nxt] = cur
                heapq.heappush(pq, (nd, nxt))

    return farthest, dist, prev


def longest_path(mask):
    points = np.argwhere(mask)
    if points.shape[0] == 0:
        return [], (None, None)
    if points.shape[0] == 1:
        p = (int(points[0, 0]), int(points[0, 1]))
        return [p], (p, p)

    seed = (int(points[0, 0]), int(points[0, 1]))
    a, _, _ = dijkstra_farthest(mask, seed, need_prev=False)
    b, _, prev = dijkstra_farthest(mask, a, need_prev=True)

    path = [b]
    cur = b
    while cur != a and cur in prev:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path, (a, b)


def endpoint_pair_by_geodesic(mask, endpoints):
    if len(endpoints) == 0:
        return None, None
    if len(endpoints) == 1:
        p = (int(endpoints[0, 0]), int(endpoints[0, 1]))
        return p, p

    ep_list = [(int(y), int(x)) for y, x in endpoints]
    best_pair = (ep_list[0], ep_list[1])
    best_dist = -1.0

    for s in ep_list:
        _, dist_map, _ = dijkstra_farthest(mask, s, need_prev=False)
        for t in ep_list:
            if t == s:
                continue
            d = dist_map.get(t, -1.0)
            if d > best_dist:
                best_dist = d
                best_pair = (s, t)

    return best_pair


def branch_junctions_from_main(mask, main_path, min_branch_keep_length=0):
    path_set = set(main_path)
    off_mask = mask.copy()
    for y, x in main_path:
        off_mask[y, x] = False

    n_labels, labels = cv2.connectedComponents(off_mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return [], 0, [], set()

    junction_points = []
    branch_count = 0
    branch_lengths = []
    branch_pixels = set()
    for cid in range(1, n_labels):
        ys, xs = np.where(labels == cid)
        if ys.size == 0:
            continue

        branch_mask = labels == cid
        branch_path, _ = longest_path(branch_mask)
        branch_len = max(0, len(branch_path) - 1)
        if branch_len < min_branch_keep_length:
            continue

        for by, bx in zip(ys, xs):
            branch_pixels.add((int(by), int(bx)))

        touches = set()
        for by, bx in zip(ys, xs):
            for ny, nx, _ in iter_neighbors(mask, int(by), int(bx)):
                if (ny, nx) in path_set:
                    touches.add((ny, nx))

        if not touches:
            continue

        branch_count += 1
        branch_lengths.append(int(branch_len))
        ts = np.array(list(touches), dtype=np.float32)
        cy, cx = np.mean(ts, axis=0)
        junction_points.append((int(round(cy)), int(round(cx))))

    if not junction_points:
        return [], branch_count, branch_lengths, branch_pixels

    merged = []
    used = [False] * len(junction_points)
    for i, p in enumerate(junction_points):
        if used[i]:
            continue
        group = [p]
        used[i] = True
        for j in range(i + 1, len(junction_points)):
            if used[j]:
                continue
            q = junction_points[j]
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= 25:
                used[j] = True
                group.append(q)
        gy = int(round(np.mean([g[0] for g in group])))
        gx = int(round(np.mean([g[1] for g in group])))
        merged.append((gy, gx))

    return merged, branch_count, branch_lengths, branch_pixels


def get_screen_size():
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


def fit_to_screen(image, margin=120):
    screen_w, screen_h = get_screen_size()
    max_w = max(200, screen_w - margin)
    max_h = max(200, screen_h - margin)

    h, w = image.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale >= 1.0:
        return image, 1.0

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def process_image(image_path, output_path="result_paths.jpg"):
    """处理单张图片，返回结果路径和路径信息"""
    MAX_BRANCH_LENGTH = 6
    MIN_BRANCH_KEEP_LENGTH = 0
    CORE_CLOSED_RATIO = 0.6

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    binary = to_binary(img)
    skeleton_raw = skeletonize(binary == 255)
    skeleton = prune_short_branches(skeleton_raw, MAX_BRANCH_LENGTH)

    n_labels, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    path_count = max(0, n_labels - 1)

    image_is_closed = False
    closed_main_path = []
    if path_count == 1:
        comp_mask = labels == 1
        closed_main_path, closed_core_mask = core_path_points(comp_mask)
        core_ratio = (len(closed_main_path) / float(np.count_nonzero(comp_mask))) if np.count_nonzero(comp_mask) else 0.0
        image_is_closed = len(closed_main_path) > 0 and core_ratio >= CORE_CLOSED_RATIO

    original_panel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    annotated_panel = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    path_infos = []
    for comp_id in range(1, n_labels):
        comp_mask = labels == comp_id
        comp_points = np.argwhere(comp_mask)
        if comp_points.shape[0] == 0:
            continue

        is_closed = image_is_closed and path_count == 1
        if is_closed:
            main_path = closed_main_path
            start_pt, end_pt = None, None
        else:
            main_path, (start_pt, end_pt) = longest_path(comp_mask)

        endpoints = find_endpoints(comp_mask)
        main_color, branch_color = path_color_pair(comp_id - 1)

        junctions, branch_count, branch_lengths, branch_pixels = branch_junctions_from_main(
            comp_mask,
            main_path,
            min_branch_keep_length=MIN_BRANCH_KEEP_LENGTH,
        )

        main_path_set = {(int(y), int(x)) for y, x in main_path}
        for y, x in main_path_set:
            annotated_panel[int(y), int(x)] = main_color
        for y, x in branch_pixels:
            annotated_panel[int(y), int(x)] = branch_color

        if is_closed:
            anchor = choose_closed_anchor(comp_mask, main_path, junctions)
            start_pt, end_pt = anchor, anchor
        elif endpoints.shape[0] >= 2:
            start_pt, end_pt = endpoint_pair_by_geodesic(comp_mask, endpoints)
        elif endpoints.shape[0] == 1:
            p = (int(endpoints[0, 0]), int(endpoints[0, 1]))
            start_pt, end_pt = p, p

        longest_branch_length = max(branch_lengths) if branch_lengths else 0
        shortest_branch_length = min(branch_lengths) if branch_lengths else 0

        if start_pt is not None and end_pt is not None:
            cv2.circle(annotated_panel, (int(start_pt[1]), int(start_pt[0])), 12, (0, 0, 255), 2)
            cv2.circle(annotated_panel, (int(end_pt[1]), int(end_pt[0])), 8, (0, 0, 255), -1)

        for y, x in junctions:
            cv2.circle(annotated_panel, (int(x), int(y)), 5, (255, 0, 0), 2)

        path_infos.append(
            {
                "id": len(path_infos) + 1,
                "is_closed": bool(is_closed),
                "main_color": main_color,
                "branch_color": branch_color,
                "pixels": int(comp_points.shape[0]),
                "branch_count": int(branch_count),
                "junction_count": int(len(junctions)),
                "longest_branch_length": int(longest_branch_length),
                "shortest_branch_length": int(shortest_branch_length),
                "start": start_pt,
                "end": end_pt,
            }
        )

    cv2.putText(original_panel, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(annotated_panel, "All Independent Paths", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(annotated_panel, f"Path count: {path_count}", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        annotated_panel,
        f"Branch keep len >= {MIN_BRANCH_KEEP_LENGTH}",
        (10, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(annotated_panel, "Main path / Branch path use different colors", (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    status_text = "Closed main route" if image_is_closed else "Open multi-path"
    status_color = (0, 255, 0) if image_is_closed else (0, 165, 255)
    cv2.putText(annotated_panel, f"Status: {status_text}", (10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    gap = 20
    result = np.zeros((img.shape[0], img.shape[1] * 2 + gap, 3), dtype=np.uint8)
    result[:, :img.shape[1]] = original_panel
    result[:, img.shape[1] + gap:] = annotated_panel

    cv2.imwrite(output_path, result)

    return result, path_count, image_is_closed, path_infos, output_path


# ==================== GUI 界面 ====================
class PathAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("路径分析工具 - 选择图片")
        self.root.geometry("500x250")
        self.root.resizable(False, False)

        # 当前选中的文件路径
        self.current_image_path = None

        # 创建界面组件
        self.create_widgets()

    def create_widgets(self):
        # 标题
        title_label = tk.Label(
            self.root,
            text="图像路径分析工具",
            font=("Arial", 18, "bold"),
            fg="#2c3e50"
        )
        title_label.pack(pady=15)

        # 描述
        desc_label = tk.Label(
            self.root,
            text="选择一张二值化或灰度图像，自动提取骨骼路径并分析",
            font=("Arial", 10),
            fg="#7f8c8d"
        )
        desc_label.pack()

        # 按钮框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=25)

        # 选择图片按钮
        self.select_btn = tk.Button(
            button_frame,
            text="📁 选择图片",
            font=("Arial", 12),
            bg="#3498db",
            fg="white",
            padx=20,
            pady=10,
            command=self.select_image,
            cursor="hand2"
        )
        self.select_btn.pack(side=tk.LEFT, padx=10)

        # 退出按钮
        exit_btn = tk.Button(
            button_frame,
            text="退出",
            font=("Arial", 12),
            bg="#e74c3c",
            fg="white",
            padx=20,
            pady=10,
            command=self.root.quit,
            cursor="hand2"
        )
        exit_btn.pack(side=tk.LEFT, padx=10)

        # 文件路径显示
        self.path_label = tk.Label(
            self.root,
            text="未选择文件",
            font=("Arial", 9),
            fg="#95a5a6",
            wraplength=450
        )
        self.path_label.pack(pady=5)

        # 状态标签
        self.status_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 10),
            fg="#27ae60"
        )
        self.status_label.pack(pady=10)

        # 进度条（简单的文字提示）
        self.progress_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 9),
            fg="#f39c12"
        )
        self.progress_label.pack()

    def select_image(self):
        """打开文件选择对话框"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("PNG图像", "*.png"),
                ("JPEG图像", "*.jpg *.jpeg"),
                ("所有文件", "*.*")
            ]
        )

        if not file_path:
            return

        self.current_image_path = file_path
        self.path_label.config(text=f"已选择: {file_path}", fg="#2c3e50")
        self.status_label.config(text="⏳ 正在处理...", fg="#f39c12")
        self.select_btn.config(state=tk.DISABLED)

        # 在新线程中处理，避免界面卡顿
        thread = threading.Thread(target=self.run_analysis, daemon=True)
        thread.start()

    def run_analysis(self):
        """执行分析"""
        try:
            output_path = "result_paths.jpg"
            result, path_count, is_closed, path_infos, saved_path = process_image(
                self.current_image_path,
                output_path
            )

            # 更新界面（在主线程中执行）
            self.root.after(0, self.on_analysis_complete, path_count, is_closed, path_infos, saved_path)

        except Exception as e:
            self.root.after(0, self.on_analysis_error, str(e))

    def on_analysis_complete(self, path_count, is_closed, path_infos, saved_path):
        """分析完成回调"""
        self.select_btn.config(state=tk.NORMAL)
        self.progress_label.config(text="")

        status_text = "✅ 处理完成!"
        if is_closed:
            status_text += " (封闭路径)"
        else:
            status_text += f" (开放路径, {path_count}条)"
        self.status_label.config(text=status_text, fg="#27ae60")

        # 显示信息弹窗
        info_text = f"路径总数: {path_count}\n"
        info_text += f"整体判定: {'封闭路径' if is_closed else '开放/非封闭路径'}\n\n"
        for info in path_infos:
            path_type = '封闭' if info['is_closed'] else '开放'
            info_text += f"路径{info['id']}: {path_type}, 像素={info['pixels']}, "
            info_text += f"叉路数={info['branch_count']}, 分叉口={info['junction_count']}\n"

        info_text += f"\n结果已保存: {saved_path}"

        messagebox.showinfo("分析结果", info_text)

        # 显示图像
        try:
            display_img, scale = fit_to_screen(cv2.imread(saved_path))
            cv2.namedWindow("Path Analysis", cv2.WINDOW_NORMAL)
            cv2.imshow("Path Analysis", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            messagebox.showwarning("显示警告", f"无法显示图像: {e}")

    def on_analysis_error(self, error_msg):
        """分析出错回调"""
        self.select_btn.config(state=tk.NORMAL)
        self.progress_label.config(text="")
        self.status_label.config(text=f"❌ 错误: {error_msg}", fg="#e74c3c")
        messagebox.showerror("处理错误", f"图像处理失败:\n{error_msg}")


def main():
    root = tk.Tk()
    app = PathAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()