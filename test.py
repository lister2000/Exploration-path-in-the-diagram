"""
小路检测工具 - 三窗口版（高级路径提取算法）
- 简化版：只保留区域扫描算法
- 窗口1: 原图 + 扫描标记（蓝色标记检测区域）
- 窗口2: 二值图（黑白）
- 窗口3: 原图 + 彩色路径标记（路径与原图精确对齐）
- 显示进度百分比和预计剩余时间
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
import threading
import math
import heapq
import time
from collections import deque
import skimage.morphology as morph


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
        """右键点击 - 图片适合窗口显示"""
        self._fit_to_window()
    
    def _fit_to_window(self):
        """调整图片适合窗口显示"""
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
    
    def _on_resize(self, event):
        self.update_display()


class PathDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("小路检测工具 - 区域扫描版")
        self.root.geometry("1600x780")
        
        self.img_path = None
        self.img_color = None
        self.img_gray = None
        self.img_h = 0
        self.img_w = 0
        
        # 存储结果
        self.binary_img = None
        self.skeleton = None
        self.selected_paths = []  # 存储路径点列表 [(x, y), ...]
        self.path_colors = []
        self.path_infos = []
        
        # 缩放画布对象
        self.zoom_orig = None
        self.zoom_binary = None
        self.zoom_path = None
        
        # 是否反转
        self.is_inverted = False
        
        self._build_ui()
        self._show_status("就绪 - 请选择一张图片")
    
    def _build_ui(self):
        style = ttk.Style()
        style.configure("Scan.TButton", foreground="blue", font=("Arial", 10, "bold"))
        style.configure("Path.TButton", foreground="red", font=("Arial", 10, "bold"))
        style.configure("Invert.TButton", foreground="orange", font=("Arial", 10, "bold"))
        style.configure("Save.TButton", foreground="green", font=("Arial", 10, "bold"))
        
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(side=tk.TOP, fill=tk.X)
        
        param_frame = ttk.Frame(ctrl)
        param_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(param_frame, text="📁 选择图片", command=self._load_image).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(param_frame, text="未选择文件")
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(param_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Separator(param_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.btn_invert = ttk.Button(param_frame, text="🔄 反转颜色", command=self._invert_image)
        self.btn_invert.pack(side=tk.LEFT, padx=5)
        self.btn_invert.config(style="Invert.TButton")
        self.btn_invert.config(state="disabled")
        
        ttk.Separator(param_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(param_frame, text="灰度阈值:").pack(side=tk.LEFT, padx=(10, 2))
        self.threshold_var = tk.IntVar(value=180)
        self.threshold_label = ttk.Label(param_frame, text="180", width=4)
        self.threshold_label.pack(side=tk.LEFT)
        self.threshold_scale = tk.Scale(param_frame, from_=1, to=255, variable=self.threshold_var,
                                         orient=tk.HORIZONTAL, command=self._on_threshold_change)
        self.threshold_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        ttk.Label(param_frame, text="窗口高度:").pack(side=tk.LEFT, padx=(10, 2))
        self.min_width_var = tk.IntVar(value=20)
        self.min_width_label = ttk.Label(param_frame, text="20", width=4)
        self.min_width_label.pack(side=tk.LEFT)
        self.min_width_scale = tk.Scale(param_frame, from_=5, to=50, variable=self.min_width_var,
                                         orient=tk.HORIZONTAL, command=self._on_min_width_change)
        self.min_width_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        ttk.Label(param_frame, text="窗口宽度:").pack(side=tk.LEFT, padx=(10, 2))
        self.max_width_var = tk.IntVar(value=25)
        self.max_width_label = ttk.Label(param_frame, text="25", width=4)
        self.max_width_label.pack(side=tk.LEFT)
        self.max_width_scale = tk.Scale(param_frame, from_=5, to=100, variable=self.max_width_var,
                                         orient=tk.HORIZONTAL, command=self._on_max_width_change)
        self.max_width_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        ttk.Label(param_frame, text="去毛刺阈值:").pack(side=tk.LEFT, padx=(10, 2))
        self.prune_var = tk.IntVar(value=6)
        self.prune_label = ttk.Label(param_frame, text="6", width=4)
        self.prune_label.pack(side=tk.LEFT)
        self.prune_scale = tk.Scale(param_frame, from_=1, to=50, variable=self.prune_var,
                                     orient=tk.HORIZONTAL, command=self._on_prune_change)
        self.prune_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        ttk.Label(param_frame, text="保留路径数:").pack(side=tk.LEFT, padx=(10, 2))
        self.keep_paths_var = tk.IntVar(value=5)
        self.keep_paths_label = ttk.Label(param_frame, text="5", width=4)
        self.keep_paths_label.pack(side=tk.LEFT)
        self.keep_paths_scale = tk.Scale(param_frame, from_=1, to=20, variable=self.keep_paths_var,
                                          orient=tk.HORIZONTAL, command=self._on_keep_paths_change)
        self.keep_paths_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        self.btn_scan = ttk.Button(btn_frame, text="步骤1: 扫描生成二值图", command=self._run_scan)
        self.btn_scan.pack(side=tk.LEFT, padx=5)
        self.btn_scan.config(style="Scan.TButton")
        self.btn_scan.config(state="disabled")
        
        self.btn_save_binary = ttk.Button(btn_frame, text="💾 保存二值图", command=self._save_binary_image)
        self.btn_save_binary.pack(side=tk.LEFT, padx=5)
        self.btn_save_binary.config(style="Save.TButton")
        self.btn_save_binary.config(state="disabled")
        
        self.btn_path = ttk.Button(btn_frame, text="步骤2: 提取路径(高级)", command=self._run_extract_path)
        self.btn_path.pack(side=tk.LEFT, padx=5)
        self.btn_path.config(style="Path.TButton")
        self.btn_path.config(state="disabled")
        
        ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(btn_frame, text="🔄 重置", command=self._reset_all).pack(side=tk.LEFT, padx=5)
        
        # 进度显示（增强）
        self.progress_label = ttk.Label(btn_frame, text="", font=("Arial", 9))
        self.progress_label.pack(side=tk.LEFT, padx=10)
        
        # 时间显示
        self.time_label = ttk.Label(btn_frame, text="", font=("Arial", 9), foreground="blue")
        self.time_label.pack(side=tk.LEFT, padx=10)
        
        tip_label = ttk.Label(btn_frame, text="🖱 左键拉框放大 | 中键平移 | 右键适合窗口", 
                              font=("Arial", 9), foreground="gray")
        tip_label.pack(side=tk.RIGHT, padx=10)
        
        canvas_frame = ttk.Frame(self.root, padding=5)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        window1 = ttk.LabelFrame(canvas_frame, text="窗口1: 原图 + 扫描标记 (蓝色区域)")
        window1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.canvas_orig = tk.Canvas(window1, bg="gray")
        self.canvas_orig.pack(fill=tk.BOTH, expand=True)
        
        window2 = ttk.LabelFrame(canvas_frame, text="窗口2: 二值图 (黑底白线)")
        window2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.canvas_binary = tk.Canvas(window2, bg="gray")
        self.canvas_binary.pack(fill=tk.BOTH, expand=True)
        
        window3 = ttk.LabelFrame(canvas_frame, text="窗口3: 原图 + 彩色路径 (🟡空心=起点, 🟡实心=终点)")
        window3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.canvas_path = tk.Canvas(window3, bg="gray")
        self.canvas_path.pack(fill=tk.BOTH, expand=True)
        
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
    
    def _on_min_width_change(self, val):
        v = int(float(val))
        self.min_width_var.set(v)
        self.min_width_label.config(text=str(v))
        if v > self.max_width_var.get():
            self.max_width_var.set(v)
            self.max_width_label.config(text=str(v))
            self.max_width_scale.set(v)
    
    def _on_max_width_change(self, val):
        v = int(float(val))
        self.max_width_var.set(v)
        self.max_width_label.config(text=str(v))
        if v < self.min_width_var.get():
            self.min_width_var.set(v)
            self.min_width_label.config(text=str(v))
            self.min_width_scale.set(v)
    
    def _on_prune_change(self, val):
        v = int(float(val))
        self.prune_var.set(v)
        self.prune_label.config(text=str(v))
    
    def _on_keep_paths_change(self, val):
        v = int(float(val))
        self.keep_paths_var.set(v)
        self.keep_paths_label.config(text=str(v))
    
    def _invert_image(self):
        if self.img_color is None:
            self._show_status("请先选择图片")
            return
        
        self.is_inverted = not self.is_inverted
        
        if self.is_inverted:
            self.img_gray = cv2.bitwise_not(self.img_gray)
            self.img_color = cv2.bitwise_not(self.img_color)
            self.btn_invert.config(text="🔄 恢复颜色")
            self._show_status("已反转颜色")
        else:
            self.img_gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
            self.btn_invert.config(text="🔄 反转颜色")
            self._show_status("已恢复原始颜色")
        
        if self.zoom_orig is not None:
            self.zoom_orig.update_image(self.img_color)
        
        self._reset_scan_results()
        
        self.btn_scan.config(state="normal")
        self.btn_save_binary.config(state="disabled")
        self.btn_path.config(state="disabled")
        
        h, w = self.img_gray.shape
        black_img = np.zeros((h, w, 3), dtype=np.uint8)
        if self.zoom_binary is not None:
            self.zoom_binary.update_image(black_img)
        if self.zoom_path is not None:
            self.zoom_path.update_image(self.img_color)
        
        self.progress_label.config(text="")
        self.time_label.config(text="")
        self.info_label.config(text="")
        self._show_status(f"{'已反转' if self.is_inverted else '已恢复'}颜色，请重新执行步骤1")
    
    def _reset_scan_results(self):
        self.binary_img = None
        self.skeleton = None
        self.selected_paths = []
        self.path_colors = []
        self.path_infos = []
    
    def _reset_all(self):
        if self.img_gray is None:
            return
        
        h, w = self.img_gray.shape
        black_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self.zoom_orig is not None:
            self.zoom_orig.update_image(self.img_color)
        
        if self.zoom_binary is not None:
            self.zoom_binary.update_image(black_img)
        
        if self.zoom_path is not None:
            self.zoom_path.update_image(self.img_color)
        
        self._reset_scan_results()
        
        self.btn_scan.config(state="normal")
        self.btn_save_binary.config(state="disabled")
        self.btn_path.config(state="disabled")
        
        self.progress_label.config(text="")
        self.time_label.config(text="")
        self.info_label.config(text="")
        self._show_status("已重置")
    
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("所有文件", "*.*")]
        )
        if not path:
            return
        
        self.img_color = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.img_color is None:
            self._show_status(f"无法读取图片: {path}")
            return
        
        self.img_gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        self.img_h, self.img_w = self.img_gray.shape
        self.img_path = path
        self.is_inverted = False
        self.btn_invert.config(text="🔄 反转颜色")
        self.file_label.config(text=os.path.basename(path))
        self._show_status(f"已加载: {path}  尺寸: {self.img_w}x{self.img_h}")
        
        h, w = self.img_gray.shape
        black_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        self.zoom_orig = ZoomableCanvas(self.canvas_orig, self.img_color)
        self.zoom_binary = ZoomableCanvas(self.canvas_binary, black_img)
        self.zoom_path = ZoomableCanvas(self.canvas_path, self.img_color)
        
        self._reset_scan_results()
        
        self.btn_invert.config(state="normal")
        self.btn_scan.config(state="normal")
        self.btn_save_binary.config(state="disabled")
        self.btn_path.config(state="disabled")
        
        self.progress_label.config(text="")
        self.time_label.config(text="")
        self.info_label.config(text="")
    
    def _run_scan(self):
        if self.img_gray is None:
            self._show_status("请先选择图片")
            return
        
        self.btn_scan.config(state="disabled")
        self._show_status("扫描中... (算法: 区域扫描)")
        self.progress_label.config(text="扫描中... 0%")
        self.time_label.config(text="预计剩余: --")
        
        thread = threading.Thread(target=self._scan_thread, daemon=True)
        thread.start()
    
    def _scan_thread(self):
        try:
            start_time = time.time()
            threshold = self.threshold_var.get()
            min_width = self.min_width_var.get()
            max_width = self.max_width_var.get()
            
            h, w = self.img_gray.shape
            
            # 使用区域扫描（带进度）
            binary_img, progress_info = self._region_scan_with_progress(
                self.img_gray, threshold, min_width, max_width, start_time
            )
            
            self.binary_img = binary_img
            total_pixels = np.sum(binary_img == 255)
            
            # 计算总耗时
            elapsed = time.time() - start_time
            time_str = self._format_time(elapsed)
            
            # ===== 更新窗口1: 原图 + 标记 =====
            marked_img = self.img_color.copy()
            blue_color = np.array([255, 0, 0], dtype=np.uint8)
            marked_img[binary_img == 255] = blue_color
            
            cv2.putText(marked_img, f"算法: 区域扫描", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(marked_img, f"窗口: {min_width}x{max_width}", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            info_text = f"总计:{total_pixels}px"
            
            self.root.after(0, self._update_window1, marked_img)
            
            # ===== 更新窗口2: 二值图 =====
            binary_color = np.stack([binary_img] * 3, axis=-1)
            cv2.putText(binary_color, f"算法: 区域扫描", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(binary_color, f"像素: {total_pixels}", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            self.root.after(0, self._update_window2, binary_color)
            
            self.btn_save_binary.config(state="normal")
            self.btn_path.config(state="normal")
            self.btn_scan.config(state="normal")
            
            self.root.after(0, self._show_status, 
                           f"✅ 扫描完成 (区域扫描): {info_text} (耗时: {time_str})")
            self.root.after(0, self._update_progress, 
                           f"扫描完成! {info_text}")
            self.root.after(0, self._update_time, 
                           f"耗时: {time_str}")
            
        except Exception as e:
            self.root.after(0, self._show_status, f"❌ 扫描出错: {str(e)}")
            self.root.after(0, self._enable_scan_btn)
            import traceback
            traceback.print_exc()
    
    def _region_scan_with_progress(self, gray_img, threshold, min_width, max_width, start_time):
        """
        区域扫描算法（带进度显示）
        """
        h, w = gray_img.shape
        binary_img = np.zeros((h, w), dtype=np.uint8)
        
        rect_h = min_width
        rect_w = max_width
        
        if rect_w < 1 or rect_h < 1:
            rect_w = 20
            rect_h = 20
        
        # 使用积分图加速
        integral = np.zeros((h + 1, w + 1), dtype=np.float64)
        integral[1:, 1:] = np.cumsum(np.cumsum(gray_img.astype(np.float64), axis=0), axis=1)
        
        total_rows = h - rect_h + 1
        processed_rows = 0
        
        for y in range(h - rect_h + 1):
            for x in range(w - rect_w + 1):
                sum_val = (integral[y + rect_h, x + rect_w] - 
                           integral[y, x + rect_w] - 
                           integral[y + rect_h, x] + 
                           integral[y, x])
                area = rect_h * rect_w
                avg_val = sum_val / area
                
                if avg_val <= threshold:
                    binary_img[y:y + rect_h, x:x + rect_w] = 255
            
            processed_rows += 1
            
            # 每处理5行更新一次进度
            if processed_rows % 5 == 0 or processed_rows == total_rows:
                progress = int((processed_rows / total_rows) * 100)
                elapsed = time.time() - start_time
                
                # 估算剩余时间
                if progress > 0:
                    total_estimated = elapsed / (progress / 100)
                    remaining = total_estimated - elapsed
                    time_str = self._format_time(remaining)
                else:
                    time_str = "--"
                
                self.root.after(0, self._update_progress, f"扫描中... {progress}%")
                self.root.after(0, self._update_time, f"预计剩余: {time_str}")
        
        return binary_img, (total_rows, processed_rows)
    
    def _format_time(self, seconds):
        """格式化时间显示"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
    
    def _region_scan(self, gray_img, threshold, min_width, max_width):
        """区域扫描算法（无进度，保留用于兼容）"""
        h, w = gray_img.shape
        binary_img = np.zeros((h, w), dtype=np.uint8)
        
        rect_h = min_width
        rect_w = max_width
        
        if rect_w < 1 or rect_h < 1:
            rect_w = 20
            rect_h = 20
        
        integral = np.zeros((h + 1, w + 1), dtype=np.float64)
        integral[1:, 1:] = np.cumsum(np.cumsum(gray_img.astype(np.float64), axis=0), axis=1)
        
        for y in range(h - rect_h + 1):
            for x in range(w - rect_w + 1):
                sum_val = (integral[y + rect_h, x + rect_w] - 
                           integral[y, x + rect_w] - 
                           integral[y + rect_h, x] + 
                           integral[y, x])
                area = rect_h * rect_w
                avg_val = sum_val / area
                
                if avg_val <= threshold:
                    binary_img[y:y + rect_h, x:x + rect_w] = 255
        
        return binary_img
    
    def _save_binary_image(self):
        if self.binary_img is None:
            self._show_status("请先执行步骤1: 扫描生成二值图")
            return
        
        if self.img_path:
            directory = os.path.dirname(self.img_path)
            base_name = os.path.splitext(os.path.basename(self.img_path))[0]
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
            cv2.imwrite(file_path, self.binary_img)
            self._show_status(f"✅ 二值图已保存: {file_path}")
        except Exception as e:
            self._show_status(f"❌ 保存失败: {str(e)}")
    
    # ==================== 高级路径提取算法 ====================
    
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
    
    def _iter_neighbors(self, mask, y, x):
        h, w = mask.shape
        for dy, dx, weight in self.NEIGHBORS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                yield ny, nx, weight
    
    def _neighbor_degree(self, mask, y, x):
        return sum(1 for _ in self._iter_neighbors(mask, y, x))
    
    def _find_endpoints_advanced(self, skeleton_bool):
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
    
    def _prune_short_branches_advanced(self, skeleton_255, max_branch_length):
        skel = skeleton_255.copy().astype(bool)
        
        while True:
            endpoints = self._find_endpoints_advanced(skel)
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
    
    def _remove_branches_advanced(self, all_paths):
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
                        if self._is_connected_advanced(all_paths[idx], all_paths[j], 5):
                            connected.append(j)
                            used[j] = True
                            changed = True
                            break
            
            if connected:
                best_path = max([all_paths[idx] for idx in connected], key=len)
                merged.append(best_path)
        
        return merged
    
    def _is_connected_advanced(self, path1, path2, max_dist):
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
    
    def _run_extract_path(self):
        if self.binary_img is None:
            self._show_status("请先执行步骤1: 扫描")
            return
        
        self.btn_path.config(state="disabled")
        self._show_status("提取路径中 (高级算法)...")
        self.progress_label.config(text="提取路径中... 0%")
        self.time_label.config(text="预计剩余: --")
        
        thread = threading.Thread(target=self._extract_path_thread, daemon=True)
        thread.start()
    
    def _extract_path_thread(self):
        try:
            start_time = time.time()
            prune_threshold = self.prune_var.get()
            keep_count = self.keep_paths_var.get()
            
            use_img = self.binary_img
            
            # ===== 1. 骨骼化 =====
            self._update_progress_safe("骨骼化中... 10%")
            binary_01 = (use_img == 255).astype(np.uint8)
            skeleton_01 = morph.skeletonize(binary_01)
            skeleton_255 = (skeleton_01 * 255).astype(np.uint8)
            
            # ===== 2. 去毛刺 =====
            self._update_progress_safe(f"去毛刺中 (阈值: {prune_threshold})... 30%")
            skeleton_pruned = self._prune_short_branches_advanced(skeleton_255, prune_threshold)
            self.skeleton = skeleton_pruned
            
            # ===== 3. 提取所有连通路径 =====
            self._update_progress_safe("提取路径中... 50%")
            n_labels, labels = cv2.connectedComponents(skeleton_pruned.astype(np.uint8), connectivity=8)
            path_count = max(0, n_labels - 1)
            
            self._update_progress_safe(f"找到 {path_count} 条路径，使用Dijkstra提取最长路径... 70%")
            
            # ===== 4. 对每条路径使用Dijkstra提取最长路径 =====
            all_main_paths = []
            path_infos = []
            total_comps = n_labels - 1
            processed = 0
            
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
                
                processed += 1
                if total_comps > 0:
                    progress = 70 + int((processed / total_comps) * 20)
                    self._update_progress_safe(f"提取路径中... {progress}%")
                    elapsed = time.time() - start_time
                    if progress > 0:
                        total_estimated = elapsed / ((progress - 10) / 100) if progress > 10 else 0
                        remaining = total_estimated - elapsed
                        if remaining > 0:
                            time_str = self._format_time(remaining)
                            self._update_time_safe(f"预计剩余: {time_str}")
            
            # ===== 5. 去分叉 =====
            self._update_progress_safe("去分叉中... 90%")
            merged_paths = self._remove_branches_advanced(all_main_paths)
            
            # ===== 6. 按长度排序，保留前N条 =====
            if len(merged_paths) > 0:
                merged_paths.sort(key=len, reverse=True)
                self.selected_paths = merged_paths[:keep_count]
            else:
                self.selected_paths = []
            
            self.path_infos = path_infos
            
            # ===== 7. 生成路径颜色 =====
            self.path_colors = self._generate_colors(len(self.selected_paths))
            
            # ===== 8. 绘制结果 =====
            self._update_progress_safe("绘制结果... 95%")
            path_img = self.img_color.copy()
            self._draw_paths_on_image_advanced(path_img, self.selected_paths, self.path_colors)
            
            elapsed = time.time() - start_time
            time_str = self._format_time(elapsed)
            
            self.root.after(0, self._update_window3, path_img)
            
            # 统计信息
            path_info = f"保留 {len(self.selected_paths)} 条路径"
            if len(self.selected_paths) > 0:
                path_info += f", 最长: {len(self.selected_paths[0])}点"
            
            self.root.after(0, self._show_status, f"✅ 路径提取完成: {path_info} (耗时: {time_str})")
            self.root.after(0, self._update_progress, f"提取完成! {path_info}")
            self.root.after(0, self._update_time, f"耗时: {time_str}")
            self.root.after(0, self._update_info, path_info)
            
            self.btn_path.config(state="normal")
            
        except Exception as e:
            self.root.after(0, self._show_status, f"❌ 路径提取出错: {str(e)}")
            self.root.after(0, self._enable_path_btn)
            import traceback
            traceback.print_exc()
    
    def _update_progress_safe(self, text):
        """安全更新进度（从子线程调用）"""
        self.root.after(0, lambda: self.progress_label.config(text=text))
    
    def _update_time_safe(self, text):
        """安全更新时间（从子线程调用）"""
        self.root.after(0, lambda: self.time_label.config(text=text))
    
    def _generate_colors(self, count):
        colors = [
            (0, 0, 255),      # 红色
            (0, 165, 255),    # 橙色
            (0, 255, 255),    # 黄色
            (255, 0, 255),    # 品红
            (255, 255, 0),    # 青色
            (128, 0, 255),    # 紫色
            (0, 200, 0),      # 绿色
            (255, 128, 0),    # 蓝绿色
        ]
        
        result = []
        for i in range(count):
            result.append(colors[i % len(colors)])
        return result
    
    def _draw_paths_on_image_advanced(self, img, paths, colors):
        """
        在图像上绘制路径 - 路径坐标与原图精确对齐
        """
        h, w = img.shape[:2]
        min_width = self.min_width_var.get()
        
        # 计算绘制参数（基于最小宽度）
        line_thickness = max(1, min_width // 2)
        circle_radius = max(2, min_width // 3)
        circle_thickness = max(1, min_width // 5)
        font_scale = max(0.3, min_width / 30)
        font_thickness = max(1, min_width // 5)
        
        for idx, path in enumerate(paths):
            if len(path) < 2:
                continue
            
            color = colors[idx % len(colors)]
            
            # 确保路径点在图像范围内
            valid_path = []
            for pt in path:
                x, y = pt
                if 0 <= x < w and 0 <= y < h:
                    valid_path.append((x, y))
            
            if len(valid_path) < 2:
                continue
            
            # 绘制路径线条
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
    
    def _update_window1(self, img):
        if self.zoom_orig is not None:
            self.zoom_orig.update_image(img)
    
    def _update_window2(self, img):
        if self.zoom_binary is not None:
            self.zoom_binary.update_image(img)
    
    def _update_window3(self, img):
        if self.zoom_path is not None:
            self.zoom_path.update_image(img)
    
    def _update_progress(self, text):
        self.progress_label.config(text=text)
    
    def _update_time(self, text):
        self.time_label.config(text=text)
    
    def _update_info(self, text):
        self.info_label.config(text=f"路径信息: {text}")
    
    def _enable_scan_btn(self):
        self.btn_scan.config(state="normal")
    
    def _enable_path_btn(self):
        self.btn_path.config(state="normal")
    
    def _show_status(self, message):
        self.root.after(0, lambda: self.status.config(text=message))


def main():
    root = tk.Tk()
    app = PathDetector(root)
    root.mainloop()


if __name__ == "__main__":
    main()