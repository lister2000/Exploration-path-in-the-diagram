"""
FindPath GUI 测试工具
- 选择图片
- 参数滑块调节
- 原图显示路径
- 保存二值图
- 显示实时进度百分比和所需时间
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
import threading
import time
import skimage.morphology as morph  # 添加导入
from FindPath import FindPath


class ZoomableCanvas:
    """可缩放画布类"""
    def __init__(self, canvas, image_cv):
        self.canvas = canvas
        self.image_cv = image_cv
        self.image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(self.image_rgb)
        self.original_image = self.pil_image.copy()
        
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.min_zoom = 0.05
        self.max_zoom = 20.0
        
        self.photo_image = None
        
        self.rect_start_x = None
        self.rect_start_y = None
        self.rect_id = None
        self.is_dragging_rect = False
        
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        
        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<Button-2>", self._on_middle_down)
        self.canvas.bind("<B2-Motion>", self._on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_middle_up)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Configure>", self._on_resize)
        
        self.update_display()
    
    def update_image(self, image_cv):
        self.image_cv = image_cv
        self.image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(self.image_rgb)
        self.original_image = self.pil_image.copy()
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._fit_to_window()
    
    def _fit_to_window(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        pw, ph = self.pil_image.size
        
        if canvas_width < 10 or canvas_height < 10:
            return
        
        fit_scale_x = canvas_width / pw
        fit_scale_y = canvas_height / ph
        fit_scale = min(fit_scale_x, fit_scale_y, 1.0)
        
        self.zoom_level = fit_scale
        self.offset_x = 0
        self.offset_y = 0
        self.update_display()
    
    def update_display(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            return
        
        pw, ph = self.pil_image.size
        display_w = pw * self.zoom_level
        display_h = ph * self.zoom_level
        
        if display_w < canvas_width and display_h < canvas_height:
            offset_x = (canvas_width - display_w) / 2
            offset_y = (canvas_height - display_h) / 2
        else:
            offset_x = self.offset_x
            offset_y = self.offset_y
            max_off_x = max(0, (display_w - canvas_width) / 2)
            max_off_y = max(0, (display_h - canvas_height) / 2)
            offset_x = max(-max_off_x, min(max_off_x, offset_x))
            offset_y = max(-max_off_y, min(max_off_y, offset_y))
            self.offset_x = offset_x
            self.offset_y = offset_y
        
        if self.zoom_level != 1.0 or display_w != pw or display_h != ph:
            resized = self.pil_image.resize(
                (max(1, int(display_w)), max(1, int(display_h))), 
                Image.LANCZOS
            )
        else:
            resized = self.pil_image
        
        self.photo_image = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2 + offset_x,
            canvas_height // 2 + offset_y,
            image=self.photo_image,
            anchor=tk.CENTER
        )
        
        if self.rect_id is not None and self.rect_start_x is not None:
            self.rect_id = self.canvas.create_rectangle(
                self.rect_start_x, self.rect_start_y,
                self.rect_end_x, self.rect_end_y,
                outline="yellow", width=2, dash=(5, 3)
            )
    
    def _on_left_down(self, event):
        self.rect_start_x = event.x
        self.rect_start_y = event.y
        self.rect_end_x = event.x
        self.rect_end_y = event.y
        self.is_dragging_rect = True
        self.rect_id = self.canvas.create_rectangle(
            self.rect_start_x, self.rect_start_y,
            self.rect_end_x, self.rect_end_y,
            outline="yellow", width=2, dash=(5, 3)
        )
    
    def _on_left_drag(self, event):
        if not self.is_dragging_rect:
            return
        self.rect_end_x = event.x
        self.rect_end_y = event.y
        if self.rect_id is not None:
            self.canvas.coords(
                self.rect_id,
                self.rect_start_x, self.rect_start_y,
                self.rect_end_x, self.rect_end_y
            )
    
    def _on_left_up(self, event):
        if not self.is_dragging_rect:
            return
        self.is_dragging_rect = False
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        
        x1 = min(self.rect_start_x, self.rect_end_x)
        y1 = min(self.rect_start_y, self.rect_end_y)
        x2 = max(self.rect_start_x, self.rect_end_x)
        y2 = max(self.rect_start_y, self.rect_end_y)
        
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            self.rect_start_x = None
            self.rect_start_y = None
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        pw, ph = self.pil_image.size
        
        display_w = pw * self.zoom_level
        display_h = ph * self.zoom_level
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        img_center_x = (center_x - canvas_width/2 - self.offset_x) / display_w * pw
        img_center_y = (center_y - canvas_height/2 - self.offset_y) / display_h * ph
        
        rect_w = abs(x2 - x1)
        rect_h = abs(y2 - y1)
        scale_x = canvas_width / rect_w
        scale_y = canvas_height / rect_h
        new_zoom = min(scale_x, scale_y) * self.zoom_level * 0.95
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        self.zoom_level = new_zoom
        new_display_w = pw * self.zoom_level
        new_display_h = ph * self.zoom_level
        self.offset_x = canvas_width/2 - (img_center_x / pw) * new_display_w
        self.offset_y = canvas_height/2 - (img_center_y / ph) * new_display_h
        
        self.rect_start_x = None
        self.rect_start_y = None
        self.update_display()
    
    def _on_middle_down(self, event):
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.pan_offset_x = self.offset_x
        self.pan_offset_y = self.offset_y
        self.canvas.config(cursor="fleur")
    
    def _on_middle_drag(self, event):
        if not self.is_panning:
            return
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.offset_x = self.pan_offset_x + dx
        self.offset_y = self.pan_offset_y + dy
        self.update_display()
    
    def _on_middle_up(self, event):
        self.is_panning = False
        self.canvas.config(cursor="")
    
    def _on_right_click(self, event):
        self._fit_to_window()
    
    def _on_resize(self, event):
        self.update_display()


# ==================== 带进度回调的FindPath包装类 ====================

class FindPathWithProgress:
    """带进度回调的FindPath包装类"""
    
    def __init__(self, image, threshold=180, window_h=20, window_w=25,
                 prune_threshold=6, keep_paths=5, invert=False, 
                 progress_callback=None, verbose=False):
        
        self.finder = FindPath(
            image=image,
            threshold=threshold,
            window_h=window_h,
            window_w=window_w,
            prune_threshold=prune_threshold,
            keep_paths=keep_paths,
            invert=invert,
            verbose=verbose
        )
        self.progress_callback = progress_callback
        self.start_time = None
        
        # 重写finder的_region_scan方法
        self.finder._region_scan = self._region_scan_with_progress
        self.finder._extract_paths = self._extract_paths_with_progress
    
    def _region_scan_with_progress(self):
        """带进度的区域扫描"""
        if self.progress_callback:
            self.progress_callback("区域扫描开始...", 10)
        
        h, w = self.finder.img_h, self.finder.img_w
        binary_img = np.zeros((h, w), dtype=np.uint8)
        
        rect_h = self.finder.window_h
        rect_w = self.finder.window_w
        
        if rect_w < 1 or rect_h < 1:
            rect_w = 20
            rect_h = 20
        
        # 使用积分图加速
        integral = np.zeros((h + 1, w + 1), dtype=np.float64)
        integral[1:, 1:] = np.cumsum(np.cumsum(self.finder.img_gray.astype(np.float64), axis=0), axis=1)
        
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
                
                if avg_val <= self.finder.threshold:
                    binary_img[y:y + rect_h, x:x + rect_w] = 255
            
            processed += 1
            
            # 每处理5行更新一次进度
            if processed % 5 == 0 or processed == total_rows:
                progress = int((processed / total_rows) * 30) + 10
                if self.progress_callback:
                    self.progress_callback(f"区域扫描中... {progress}%", progress)
        
        if self.finder.invert:
            binary_img = cv2.bitwise_not(binary_img)
        
        if self.progress_callback:
            self.progress_callback("区域扫描完成", 40)
        
        return binary_img
    
    def _extract_paths_with_progress(self):
        """带进度的路径提取"""
        if self.progress_callback:
            self.progress_callback("骨骼化中... 45%", 45)
        
        # 1. 骨骼化
        binary_01 = (self.finder.binary_image == 255).astype(np.uint8)
        skeleton_01 = morph.skeletonize(binary_01)
        skeleton_255 = (skeleton_01 * 255).astype(np.uint8)
        
        if self.progress_callback:
            self.progress_callback("去毛刺中... 55%", 55)
        
        # 2. 去毛刺
        skeleton_pruned = self.finder._prune_short_branches(skeleton_255, self.finder.prune_threshold)
        self.finder.skeleton = skeleton_pruned
        
        if self.progress_callback:
            self.progress_callback("提取连通路径... 65%", 65)
        
        # 3. 提取所有连通路径
        n_labels, labels = cv2.connectedComponents(skeleton_pruned.astype(np.uint8), connectivity=8)
        
        # 4. 对每条路径使用Dijkstra提取最长路径
        all_main_paths = []
        path_infos = []
        total_comps = n_labels - 1
        processed = 0
        
        if self.progress_callback:
            self.progress_callback(f"使用Dijkstra提取路径 (共{total_comps}条)... 70%", 70)
        
        for comp_id in range(1, n_labels):
            comp_mask = labels == comp_id
            comp_points = np.argwhere(comp_mask)
            if comp_points.shape[0] == 0:
                continue
            
            # 使用Dijkstra算法找最长路径
            main_path, (start_pt, end_pt) = self.finder._longest_path_dijkstra(comp_mask)
            
            # 转换路径点格式: (y, x) -> (x, y) 用于绘制
            converted_path = []
            for pt in main_path:
                y, x = pt
                if 0 <= x < self.finder.img_w and 0 <= y < self.finder.img_h:
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
            
            processed += 1
            if total_comps > 0 and processed % max(1, total_comps // 10) == 0:
                progress = 70 + int((processed / total_comps) * 20)
                if self.progress_callback:
                    self.progress_callback(f"Dijkstra提取中... {progress}%", progress)
        
        if self.progress_callback:
            self.progress_callback("去分叉中... 90%", 90)
        
        # 5. 去分叉
        merged_paths = self.finder._remove_branches(all_main_paths)
        
        # 6. 按长度排序，保留前N条
        if len(merged_paths) > 0:
            merged_paths.sort(key=len, reverse=True)
            self.finder.paths = merged_paths[:self.finder.keep_paths]
        else:
            self.finder.paths = []
        
        self.finder.path_infos = path_infos
        self.finder.path_count = len(self.finder.paths)
        
        if self.progress_callback:
            self.progress_callback("绘制结果... 95%", 95)
        
        # 7. 生成路径标记图
        self.finder.path_image = self.finder._draw_paths()
        
        if self.progress_callback:
            self.progress_callback("完成! 100%", 100)
    
    def process(self):
        """执行完整的路径检测流程"""
        self.start_time = time.time()
        
        if self.progress_callback:
            self.progress_callback("初始化... 0%", 0)
        
        # 步骤1: 区域扫描生成二值图
        self.finder.binary_image = self._region_scan_with_progress()
        
        # 步骤2: 提取路径
        self._extract_paths_with_progress()
        
        elapsed = time.time() - self.start_time
        self.finder._processed = True
        
        return self.finder


# ==================== GUI主类 ====================

class FindPathGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FindPath 测试工具")
        self.root.geometry("1400x800")
        
        self.image_path = None
        self.original_image = None
        self.result_image = None
        self.binary_image = None
        self.finder = None
        self.is_running = False
        
        self._build_ui()
        self._show_status("就绪 - 请选择一张图片")
    
    def _build_ui(self):
        # 顶部控制区
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(side=tk.TOP, fill=tk.X)
        
        # 第一行：文件选择和参数
        param_frame = ttk.Frame(ctrl)
        param_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 选择图片按钮
        ttk.Button(param_frame, text="📁 选择图片", command=self._load_image).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(param_frame, text="未选择文件")
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(param_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # 灰度阈值
        ttk.Label(param_frame, text="灰度阈值:").pack(side=tk.LEFT, padx=(10, 2))
        self.threshold_var = tk.IntVar(value=180)
        self.threshold_label = ttk.Label(param_frame, text="180", width=4)
        self.threshold_label.pack(side=tk.LEFT)
        self.threshold_scale = tk.Scale(param_frame, from_=1, to=255, variable=self.threshold_var,
                                         orient=tk.HORIZONTAL, command=self._on_threshold_change)
        self.threshold_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        # 窗口高度
        ttk.Label(param_frame, text="窗口高度:").pack(side=tk.LEFT, padx=(10, 2))
        self.window_h_var = tk.IntVar(value=20)
        self.window_h_label = ttk.Label(param_frame, text="20", width=4)
        self.window_h_label.pack(side=tk.LEFT)
        self.window_h_scale = tk.Scale(param_frame, from_=5, to=50, variable=self.window_h_var,
                                        orient=tk.HORIZONTAL, command=self._on_window_h_change)
        self.window_h_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        # 窗口宽度
        ttk.Label(param_frame, text="窗口宽度:").pack(side=tk.LEFT, padx=(10, 2))
        self.window_w_var = tk.IntVar(value=25)
        self.window_w_label = ttk.Label(param_frame, text="25", width=4)
        self.window_w_label.pack(side=tk.LEFT)
        self.window_w_scale = tk.Scale(param_frame, from_=5, to=100, variable=self.window_w_var,
                                        orient=tk.HORIZONTAL, command=self._on_window_w_change)
        self.window_w_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        # 去毛刺阈值
        ttk.Label(param_frame, text="去毛刺:").pack(side=tk.LEFT, padx=(10, 2))
        self.prune_var = tk.IntVar(value=6)
        self.prune_label = ttk.Label(param_frame, text="6", width=4)
        self.prune_label.pack(side=tk.LEFT)
        self.prune_scale = tk.Scale(param_frame, from_=1, to=50, variable=self.prune_var,
                                     orient=tk.HORIZONTAL, command=self._on_prune_change)
        self.prune_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        # 保留路径数
        ttk.Label(param_frame, text="保留路径:").pack(side=tk.LEFT, padx=(10, 2))
        self.keep_var = tk.IntVar(value=5)
        self.keep_label = ttk.Label(param_frame, text="5", width=4)
        self.keep_label.pack(side=tk.LEFT)
        self.keep_scale = tk.Scale(param_frame, from_=1, to=20, variable=self.keep_var,
                                    orient=tk.HORIZONTAL, command=self._on_keep_change)
        self.keep_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        # 第二行：操作按钮
        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        # 运行按钮
        self.btn_run = ttk.Button(btn_frame, text="▶ 运行检测", command=self._run_detection)
        self.btn_run.pack(side=tk.LEFT, padx=5)
        self.btn_run.config(state="disabled")
        
        # 保存二值图按钮
        self.btn_save_binary = ttk.Button(btn_frame, text="💾 保存二值图", command=self._save_binary)
        self.btn_save_binary.pack(side=tk.LEFT, padx=5)
        self.btn_save_binary.config(state="disabled")
        
        # 保存路径图按钮
        self.btn_save_path = ttk.Button(btn_frame, text="💾 保存路径图", command=self._save_path)
        self.btn_save_path.pack(side=tk.LEFT, padx=5)
        self.btn_save_path.config(state="disabled")
        
        ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # 重置按钮
        ttk.Button(btn_frame, text="🔄 重置", command=self._reset).pack(side=tk.LEFT, padx=5)
        
        # 进度显示
        self.progress_label = ttk.Label(btn_frame, text="", font=("Arial", 9), foreground="blue")
        self.progress_label.pack(side=tk.LEFT, padx=10)
        
        # 时间显示
        self.time_label = ttk.Label(btn_frame, text="", font=("Arial", 9), foreground="green")
        self.time_label.pack(side=tk.LEFT, padx=10)
        
        # 路径数量显示
        self.count_label = ttk.Label(btn_frame, text="", font=("Arial", 9), foreground="red")
        self.count_label.pack(side=tk.LEFT, padx=10)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(btn_frame, length=150, mode='determinate', maximum=100)
        self.progress_bar.pack(side=tk.LEFT, padx=10)
        
        # 图片显示区
        canvas_frame = ttk.Frame(self.root, padding=5)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 单窗口显示
        window = ttk.LabelFrame(canvas_frame, text="检测结果 (红色路径 | 🟡空心=起点, 🟡实心=终点)")
        window.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.canvas = tk.Canvas(window, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 缩放画布对象
        self.zoom_canvas = None
        
        # 底部状态栏
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status = ttk.Label(status_frame, text="就绪", relief=tk.SUNKEN)
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.info_label = ttk.Label(status_frame, text="", font=("Arial", 9))
        self.info_label.pack(side=tk.RIGHT, padx=10, pady=2)
    
    def _on_threshold_change(self, val):
        v = int(float(val))
        self.threshold_var.set(v)
        self.threshold_label.config(text=str(v))
    
    def _on_window_h_change(self, val):
        v = int(float(val))
        self.window_h_var.set(v)
        self.window_h_label.config(text=str(v))
    
    def _on_window_w_change(self, val):
        v = int(float(val))
        self.window_w_var.set(v)
        self.window_w_label.config(text=str(v))
    
    def _on_prune_change(self, val):
        v = int(float(val))
        self.prune_var.set(v)
        self.prune_label.config(text=str(v))
    
    def _on_keep_change(self, val):
        v = int(float(val))
        self.keep_var.set(v)
        self.keep_label.config(text=str(v))
    
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("所有文件", "*.*")]
        )
        if not path:
            return
        
        self.image_path = path
        self.original_image = cv2.imread(path)
        if self.original_image is None:
            self._show_status(f"无法读取图片: {path}")
            return
        
        self.file_label.config(text=os.path.basename(path))
        self._show_status(f"已加载: {path}  尺寸: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
        
        # 显示原图
        self.result_image = self.original_image.copy()
        self.zoom_canvas = ZoomableCanvas(self.canvas, self.result_image)
        
        # 启用运行按钮
        self.btn_run.config(state="normal")
        self.btn_save_binary.config(state="disabled")
        self.btn_save_path.config(state="disabled")
        
        self.progress_label.config(text="")
        self.time_label.config(text="")
        self.count_label.config(text="")
        self.info_label.config(text="")
        self.progress_bar['value'] = 0
    
    def _update_progress(self, message, percent):
        """更新进度（从子线程调用）"""
        self.progress_label.config(text=message)
        self.progress_bar['value'] = percent
        self.root.update_idletasks()
    
    def _update_time(self, elapsed):
        """更新时间显示"""
        if elapsed < 1:
            time_str = f"{elapsed*1000:.0f}ms"
        elif elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            minutes = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{minutes}m {secs:.0f}s"
        self.time_label.config(text=f"⏱ 耗时: {time_str}")
    
    def _run_detection(self):
        if self.original_image is None:
            self._show_status("请先选择图片")
            return
        
        if self.is_running:
            return
        
        self.is_running = True
        self.btn_run.config(state="disabled")
        self.btn_save_binary.config(state="disabled")
        self.btn_save_path.config(state="disabled")
        self._show_status("运行检测中...")
        self.progress_label.config(text="初始化... 0%")
        self.time_label.config(text="⏱ 耗时: --")
        self.progress_bar['value'] = 0
        
        thread = threading.Thread(target=self._detection_thread, daemon=True)
        thread.start()
    
    def _detection_thread(self):
        try:
            start_time = time.time()
            threshold = self.threshold_var.get()
            window_h = self.window_h_var.get()
            window_w = self.window_w_var.get()
            prune = self.prune_var.get()
            keep = self.keep_var.get()
            
            # 创建进度回调
            def progress_callback(message, percent):
                self.root.after(0, self._update_progress, message, percent)
                elapsed = time.time() - start_time
                self.root.after(0, self._update_time, elapsed)
            
            # 创建带进度回调的FindPath
            finder_with_progress = FindPathWithProgress(
                image=self.original_image,
                threshold=threshold,
                window_h=window_h,
                window_w=window_w,
                prune_threshold=prune,
                keep_paths=keep,
                progress_callback=progress_callback
            )
            
            # 执行检测
            self.finder = finder_with_progress.process()
            
            # 获取结果
            self.binary_image = self.finder.binary_image
            self.result_image = self.finder.path_image
            count = self.finder.path_count
            paths = self.finder.paths
            
            # 计算总耗时
            elapsed = time.time() - start_time
            
            # 更新显示
            self.root.after(0, self._update_display, self.result_image)
            
            # 更新状态
            self.root.after(0, self._update_result_info, count, paths)
            
            self.root.after(0, self._show_status, f"✅ 检测完成! 找到 {count} 条路径")
            self.root.after(0, self._update_progress, f"✅ 完成! 找到 {count} 条路径", 100)
            
            # 启用保存按钮
            self.root.after(0, self._enable_save_buttons)
            
            self.is_running = False
            
        except Exception as e:
            self.root.after(0, self._show_status, f"❌ 检测出错: {str(e)}")
            self.root.after(0, self._update_progress, f"❌ 错误: {str(e)}", 0)
            self.root.after(0, self._enable_run_btn)
            self.is_running = False
            import traceback
            traceback.print_exc()
    
    def _update_display(self, img):
        if self.zoom_canvas is not None:
            self.zoom_canvas.update_image(img)
    
    def _update_result_info(self, count, paths):
        self.count_label.config(text=f"📊 路径: {count} 条")
        info_text = f"点数: {[len(p) for p in paths[:5]]}"
        if len(paths) > 5:
            info_text += f" ... (共{len(paths)}条)"
        self.info_label.config(text=info_text)
    
    def _enable_save_buttons(self):
        self.btn_save_binary.config(state="normal")
        self.btn_save_path.config(state="normal")
        self.btn_run.config(state="normal")
    
    def _enable_run_btn(self):
        self.btn_run.config(state="normal")
    
    def _save_binary(self):
        if self.binary_image is None:
            self._show_status("没有二值图可保存")
            return
        
        if self.image_path:
            directory = os.path.dirname(self.image_path)
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            default_name = f"{base_name}_binary.png"
        else:
            directory = "."
            default_name = "binary.png"
        
        file_path = filedialog.asksaveasfilename(
            title="保存二值图",
            initialdir=directory,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG图片", "*.png"), ("JPEG图片", "*.jpg"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            cv2.imwrite(file_path, self.binary_image)
            self._show_status(f"✅ 二值图已保存: {file_path}")
        except Exception as e:
            self._show_status(f"❌ 保存失败: {str(e)}")
    
    def _save_path(self):
        if self.result_image is None:
            self._show_status("没有路径图可保存")
            return
        
        if self.image_path:
            directory = os.path.dirname(self.image_path)
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            default_name = f"{base_name}_paths.png"
        else:
            directory = "."
            default_name = "paths.png"
        
        file_path = filedialog.asksaveasfilename(
            title="保存路径图",
            initialdir=directory,
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG图片", "*.png"), ("JPEG图片", "*.jpg"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            cv2.imwrite(file_path, self.result_image)
            self._show_status(f"✅ 路径图已保存: {file_path}")
        except Exception as e:
            self._show_status(f"❌ 保存失败: {str(e)}")
    
    def _reset(self):
        if self.original_image is not None:
            self.result_image = self.original_image.copy()
            if self.zoom_canvas is not None:
                self.zoom_canvas.update_image(self.result_image)
        
        self.binary_image = None
        self.finder = None
        self.is_running = False
        self.btn_save_binary.config(state="disabled")
        self.btn_save_path.config(state="disabled")
        self.btn_run.config(state="normal" if self.original_image is not None else "disabled")
        self.progress_label.config(text="")
        self.time_label.config(text="")
        self.count_label.config(text="")
        self.info_label.config(text="")
        self.progress_bar['value'] = 0
        self._show_status("已重置")
    
    def _show_status(self, message):
        self.root.after(0, lambda: self.status.config(text=message))


def main():
    root = tk.Tk()
    app = FindPathGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()