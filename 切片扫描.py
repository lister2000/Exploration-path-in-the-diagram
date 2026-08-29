"""
小路检测工具 - 三窗口版（骨骼化路径提取）
- 窗口1: 原图 + 扫描标记（蓝=竖向，绿=横向）
- 窗口2: 二值图（黑白）+ 膨胀后的图
- 窗口3: 原图 + 彩色路径标记
- 可调节保留路径数量
- 黄色空心点标记起点，黄色实心点标记终点
- 重新选择图片自动重置
- 右键：图片适合窗口显示
- 新增：反转颜色后扫描
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
import threading
import math
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
        self.root.title("小路检测工具 - 三窗口版")
        self.root.geometry("1600x750")
        
        self.img_path = None
        self.img_color = None
        self.img_gray = None
        self.img_gray_inverted = None   # 反转后的灰度图
        
        # 存储结果
        self.binary_img = None          # 二值图（黑色背景，白色标记）
        self.vertical_binary = None     # 竖向扫描二值图
        self.horizontal_binary = None   # 横向扫描二值图
        self.dilated_binary = None      # 膨胀后的二值图
        self.skeleton = None            # 骨骼化结果
        self.selected_paths = []        # 选中的路径列表
        self.path_colors = []           # 路径颜色列表
        
        # 缩放画布对象
        self.zoom_orig = None
        self.zoom_binary = None
        self.zoom_path = None
        
        # 是否反转
        self.is_inverted = False
        
        self._build_ui()
        self._show_status("就绪 - 请选择一张图片")
    
    def _build_ui(self):
        # 创建样式
        style = ttk.Style()
        style.configure("Scan.TButton", foreground="blue", font=("Arial", 10, "bold"))
        style.configure("Dilate.TButton", foreground="purple", font=("Arial", 10, "bold"))
        style.configure("Path.TButton", foreground="red", font=("Arial", 10, "bold"))
        style.configure("Invert.TButton", foreground="orange", font=("Arial", 10, "bold"))
        
        # 顶部控制区
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(side=tk.TOP, fill=tk.X)
        
        # 第一行：文件选择和参数
        param_frame = ttk.Frame(ctrl)
        param_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 文件选择按钮
        self.btn_load = ttk.Button(param_frame, text="📁 选择图片", command=self._load_image)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(param_frame, text="未选择文件")
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(param_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # 反转颜色按钮（新增）
        self.btn_invert = ttk.Button(param_frame, text="🔄 反转颜色", command=self._invert_image)
        self.btn_invert.pack(side=tk.LEFT, padx=5)
        self.btn_invert.config(style="Invert.TButton")
        self.btn_invert.config(state="disabled")
        
        ttk.Separator(param_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # 灰度阈值
        ttk.Label(param_frame, text="灰度阈值:").pack(side=tk.LEFT, padx=(10, 2))
        self.threshold_var = tk.IntVar(value=180)
        self.threshold_label = ttk.Label(param_frame, text="180", width=4)
        self.threshold_label.pack(side=tk.LEFT)
        self.threshold_scale = tk.Scale(param_frame, from_=1, to=255, variable=self.threshold_var,
                                         orient=tk.HORIZONTAL, command=self._on_threshold_change)
        self.threshold_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        # 最小宽度
        ttk.Label(param_frame, text="最小宽度:").pack(side=tk.LEFT, padx=(10, 2))
        self.min_width_var = tk.IntVar(value=20)
        self.min_width_label = ttk.Label(param_frame, text="20", width=4)
        self.min_width_label.pack(side=tk.LEFT)
        self.min_width_scale = tk.Scale(param_frame, from_=5, to=50, variable=self.min_width_var,
                                         orient=tk.HORIZONTAL, command=self._on_min_width_change)
        self.min_width_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        # 最大宽度
        ttk.Label(param_frame, text="最大宽度:").pack(side=tk.LEFT, padx=(10, 2))
        self.max_width_var = tk.IntVar(value=25)
        self.max_width_label = ttk.Label(param_frame, text="25", width=4)
        self.max_width_label.pack(side=tk.LEFT)
        self.max_width_scale = tk.Scale(param_frame, from_=5, to=50, variable=self.max_width_var,
                                         orient=tk.HORIZONTAL, command=self._on_max_width_change)
        self.max_width_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        # 膨胀核大小
        ttk.Label(param_frame, text="膨胀核:").pack(side=tk.LEFT, padx=(10, 2))
        self.kernel_var = tk.IntVar(value=3)
        self.kernel_label = ttk.Label(param_frame, text="3", width=4)
        self.kernel_label.pack(side=tk.LEFT)
        self.kernel_scale = tk.Scale(param_frame, from_=1, to=20, variable=self.kernel_var,
                                      orient=tk.HORIZONTAL, command=self._on_kernel_change)
        self.kernel_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        # 去毛刺阈值
        ttk.Label(param_frame, text="去毛刺阈值:").pack(side=tk.LEFT, padx=(10, 2))
        self.prune_var = tk.IntVar(value=6)
        self.prune_label = ttk.Label(param_frame, text="6", width=4)
        self.prune_label.pack(side=tk.LEFT)
        self.prune_scale = tk.Scale(param_frame, from_=1, to=50, variable=self.prune_var,
                                     orient=tk.HORIZONTAL, command=self._on_prune_change)
        self.prune_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        # 保留路径数量
        ttk.Label(param_frame, text="保留路径数:").pack(side=tk.LEFT, padx=(10, 2))
        self.keep_paths_var = tk.IntVar(value=5)
        self.keep_paths_label = ttk.Label(param_frame, text="5", width=4)
        self.keep_paths_label.pack(side=tk.LEFT)
        self.keep_paths_scale = tk.Scale(param_frame, from_=1, to=20, variable=self.keep_paths_var,
                                          orient=tk.HORIZONTAL, command=self._on_keep_paths_change)
        self.keep_paths_scale.pack(side=tk.LEFT, padx=(2, 10))
        
        # 第二行：操作按钮
        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        self.btn_scan = ttk.Button(btn_frame, text="步骤1: 扫描生成二值图", command=self._run_scan)
        self.btn_scan.pack(side=tk.LEFT, padx=5)
        self.btn_scan.config(style="Scan.TButton")
        self.btn_scan.config(state="disabled")
        
        self.btn_dilate = ttk.Button(btn_frame, text="步骤2: 膨胀二值图", command=self._run_dilate)
        self.btn_dilate.pack(side=tk.LEFT, padx=5)
        self.btn_dilate.config(style="Dilate.TButton")
        self.btn_dilate.config(state="disabled")
        
        self.btn_path = ttk.Button(btn_frame, text="步骤3: 提取路径", command=self._run_extract_path)
        self.btn_path.pack(side=tk.LEFT, padx=5)
        self.btn_path.config(style="Path.TButton")
        self.btn_path.config(state="disabled")
        
        ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Button(btn_frame, text="🔄 重置", command=self._reset_all).pack(side=tk.LEFT, padx=5)
        
        self.progress_label = ttk.Label(btn_frame, text="", font=("Arial", 9))
        self.progress_label.pack(side=tk.LEFT, padx=10)
        
        tip_label = ttk.Label(btn_frame, text="🖱 左键拉框放大 | 中键平移 | 右键适合窗口", 
                              font=("Arial", 9), foreground="gray")
        tip_label.pack(side=tk.RIGHT, padx=10)
        
        # 图片显示区 - 三栏
        canvas_frame = ttk.Frame(self.root, padding=5)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 窗口1: 原图 + 扫描标记
        window1 = ttk.LabelFrame(canvas_frame, text="窗口1: 原图 + 扫描标记 (蓝=竖向, 绿=横向)")
        window1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.canvas_orig = tk.Canvas(window1, bg="gray")
        self.canvas_orig.pack(fill=tk.BOTH, expand=True)
        
        # 窗口2: 二值图 + 膨胀
        window2 = ttk.LabelFrame(canvas_frame, text="窗口2: 二值图 (黑底白线) / 膨胀后")
        window2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.canvas_binary = tk.Canvas(window2, bg="gray")
        self.canvas_binary.pack(fill=tk.BOTH, expand=True)
        
        # 窗口3: 原图 + 路径标记
        window3 = ttk.LabelFrame(canvas_frame, text="窗口3: 原图 + 彩色路径 (🟡空心=起点, 🟡实心=终点)")
        window3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
        self.canvas_path = tk.Canvas(window3, bg="gray")
        self.canvas_path.pack(fill=tk.BOTH, expand=True)
        
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
    
    def _on_kernel_change(self, val):
        v = int(float(val))
        self.kernel_var.set(v)
        self.kernel_label.config(text=str(v))
    
    def _on_prune_change(self, val):
        v = int(float(val))
        self.prune_var.set(v)
        self.prune_label.config(text=str(v))
    
    def _on_keep_paths_change(self, val):
        v = int(float(val))
        self.keep_paths_var.set(v)
        self.keep_paths_label.config(text=str(v))
    
    def _invert_image(self):
        """反转颜色并重新显示"""
        if self.img_color is None:
            self._show_status("请先选择图片")
            return
        
        self.is_inverted = not self.is_inverted
        
        if self.is_inverted:
            # 反转图像
            self.img_gray = cv2.bitwise_not(self.img_gray)
            self.img_color = cv2.bitwise_not(self.img_color)
            self.btn_invert.config(text="🔄 恢复颜色")
            self._show_status("已反转颜色")
        else:
            # 恢复原始图像
            self.img_gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
            self.btn_invert.config(text="🔄 反转颜色")
            self._show_status("已恢复原始颜色")
        
        # 更新窗口1显示
        if self.zoom_orig is not None:
            self.zoom_orig.update_image(self.img_color)
        
        # 重置扫描结果
        self._reset_scan_results()
        
        # 重置按钮状态
        self.btn_scan.config(state="normal")
        self.btn_dilate.config(state="disabled")
        self.btn_path.config(state="disabled")
        
        # 清空窗口2和窗口3
        h, w = self.img_gray.shape
        black_img = np.zeros((h, w, 3), dtype=np.uint8)
        if self.zoom_binary is not None:
            self.zoom_binary.update_image(black_img)
        if self.zoom_path is not None:
            self.zoom_path.update_image(self.img_color)
        
        self.progress_label.config(text="")
        self.info_label.config(text="")
        self._show_status(f"{'已反转' if self.is_inverted else '已恢复'}颜色，请重新执行步骤1")
    
    def _reset_scan_results(self):
        """重置扫描结果"""
        self.binary_img = None
        self.vertical_binary = None
        self.horizontal_binary = None
        self.dilated_binary = None
        self.skeleton = None
        self.selected_paths = []
        self.path_colors = []
    
    def _reset_all(self):
        """重置所有状态"""
        if self.img_gray is None:
            return
        
        h, w = self.img_gray.shape
        black_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 重置窗口1
        if self.zoom_orig is not None:
            self.zoom_orig.update_image(self.img_color)
        
        # 重置窗口2
        if self.zoom_binary is not None:
            self.zoom_binary.update_image(black_img)
        
        # 重置窗口3
        if self.zoom_path is not None:
            self.zoom_path.update_image(self.img_color)
        
        # 重置数据
        self._reset_scan_results()
        
        self.btn_scan.config(state="normal")
        self.btn_dilate.config(state="disabled")
        self.btn_path.config(state="disabled")
        
        self._show_status("已重置")
        self.progress_label.config(text="")
        self.info_label.config(text="")
    
    def _load_image(self):
        """选择并加载图片"""
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
        self.img_path = path
        self.is_inverted = False
        self.btn_invert.config(text="🔄 反转颜色")
        self.file_label.config(text=os.path.basename(path))
        self._show_status(f"已加载: {path}  尺寸: {self.img_color.shape[1]}x{self.img_color.shape[0]}")
        
        # 创建三个窗口
        h, w = self.img_gray.shape
        black_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 窗口1: 原图
        self.zoom_orig = ZoomableCanvas(self.canvas_orig, self.img_color)
        
        # 窗口2: 黑色背景
        self.zoom_binary = ZoomableCanvas(self.canvas_binary, black_img)
        
        # 窗口3: 原图
        self.zoom_path = ZoomableCanvas(self.canvas_path, self.img_color)
        
        # 重置所有数据
        self._reset_scan_results()
        
        # 启用按钮
        self.btn_invert.config(state="normal")
        self.btn_scan.config(state="normal")
        self.btn_dilate.config(state="disabled")
        self.btn_path.config(state="disabled")
        
        self.progress_label.config(text="")
        self.info_label.config(text="")
    
    def _run_scan(self):
        """步骤1: 扫描生成二值图"""
        if self.img_gray is None:
            self._show_status("请先选择图片")
            return
        
        self.btn_scan.config(state="disabled")
        self._show_status("扫描中...")
        self.progress_label.config(text="扫描中...")
        
        thread = threading.Thread(target=self._scan_thread, daemon=True)
        thread.start()
    
    def _scan_thread(self):
        try:
            threshold = self.threshold_var.get()
            min_width = self.min_width_var.get()
            max_width = self.max_width_var.get()
            
            h, w = self.img_gray.shape
            
            # 创建黑色背景的二值图
            binary_img = np.zeros((h, w), dtype=np.uint8)
            vertical_binary = np.zeros((h, w), dtype=np.uint8)
            horizontal_binary = np.zeros((h, w), dtype=np.uint8)
            
            # ===== 竖向扫描（逐列扫描） =====
            for x in range(w):
                col = self.img_gray[:, x]
                segments = self._find_segments(col, threshold)
                
                for seg in segments:
                    seg_len = seg[1] - seg[0] + 1
                    if min_width <= seg_len <= max_width:
                        top_ok = seg[0] == 0 or self.img_gray[seg[0] - 1, x] > threshold
                        bottom_ok = seg[1] == h - 1 or self.img_gray[seg[1] + 1, x] > threshold
                        
                        if top_ok and bottom_ok:
                            for y in range(seg[0], seg[1] + 1):
                                vertical_binary[y, x] = 255
                                binary_img[y, x] = 255
            
            # ===== 横向扫描（逐行扫描） =====
            for y in range(h):
                row = self.img_gray[y, :]
                segments = self._find_segments(row, threshold)
                
                for seg in segments:
                    seg_len = seg[1] - seg[0] + 1
                    if min_width <= seg_len <= max_width:
                        left_ok = seg[0] == 0 or self.img_gray[y, seg[0] - 1] > threshold
                        right_ok = seg[1] == w - 1 or self.img_gray[y, seg[1] + 1] > threshold
                        
                        if left_ok and right_ok:
                            for x in range(seg[0], seg[1] + 1):
                                horizontal_binary[y, x] = 255
                                binary_img[y, x] = 255
            
            self.binary_img = binary_img
            self.vertical_binary = vertical_binary
            self.horizontal_binary = horizontal_binary
            
            # 统计信息
            vert_pixels = np.sum(vertical_binary == 255)
            horz_pixels = np.sum(horizontal_binary == 255)
            total_pixels = np.sum(binary_img == 255)
            
            # ===== 更新窗口1: 原图 + 标记 =====
            marked_img = self.img_color.copy()
            blue_color = np.array([255, 0, 0], dtype=np.uint8)
            marked_img[vertical_binary == 255] = blue_color
            green_color = np.array([0, 255, 0], dtype=np.uint8)
            marked_img[horizontal_binary == 255] = green_color
            
            self.root.after(0, self._update_window1, marked_img)
            
            # ===== 更新窗口2: 二值图 =====
            binary_color = np.stack([binary_img] * 3, axis=-1)
            self.root.after(0, self._update_window2, binary_color)
            
            self.btn_dilate.config(state="normal")
            self.btn_scan.config(state="normal")
            
            self.root.after(0, self._show_status, 
                           f"✅ 扫描完成: 竖向{vert_pixels}px, 横向{horz_pixels}px, 总计{total_pixels}px")
            self.root.after(0, self._update_progress, 
                           f"竖向:{vert_pixels}px | 横向:{horz_pixels}px | 总计:{total_pixels}px")
            
        except Exception as e:
            self.root.after(0, self._show_status, f"❌ 扫描出错: {str(e)}")
            self.root.after(0, self._enable_scan_btn)
            import traceback
            traceback.print_exc()
    
    def _run_dilate(self):
        """步骤2: 膨胀二值图"""
        if self.binary_img is None:
            self._show_status("请先执行步骤1: 扫描")
            return
        
        self.btn_dilate.config(state="disabled")
        self._show_status("膨胀中...")
        self.progress_label.config(text="膨胀中...")
        
        thread = threading.Thread(target=self._dilate_thread, daemon=True)
        thread.start()
    
    def _dilate_thread(self):
        try:
            kernel_size = self.kernel_var.get()
            
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            dilated = cv2.dilate(self.binary_img, kernel, iterations=1)
            self.dilated_binary = dilated
            
            original_pixels = np.sum(self.binary_img == 255)
            dilated_pixels = np.sum(dilated == 255)
            
            # ===== 更新窗口2: 膨胀后的二值图 =====
            binary_color = np.stack([dilated] * 3, axis=-1)
            h, w = dilated.shape
            info_text = f"膨胀核: {kernel_size}x{kernel_size}  原: {original_pixels}  现: {dilated_pixels}"
            cv2.putText(binary_color, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            self.root.after(0, self._update_window2, binary_color)
            
            self.btn_path.config(state="normal")
            self.btn_dilate.config(state="normal")
            
            self.root.after(0, self._show_status, 
                           f"✅ 膨胀完成: 核{kernel_size}x{kernel_size}, {original_pixels}→{dilated_pixels}px")
            self.root.after(0, self._update_progress, 
                           f"膨胀: {original_pixels}→{dilated_pixels}px")
            
        except Exception as e:
            self.root.after(0, self._show_status, f"❌ 膨胀出错: {str(e)}")
            self.root.after(0, self._enable_dilate_btn)
            import traceback
            traceback.print_exc()
    
    def _run_extract_path(self):
        """步骤3: 提取路径"""
        if self.dilated_binary is None:
            self._show_status("请先执行步骤2: 膨胀")
            return
        
        self.btn_path.config(state="disabled")
        self._show_status("提取路径中...")
        self.progress_label.config(text="提取路径中...")
        
        thread = threading.Thread(target=self._extract_path_thread, daemon=True)
        thread.start()
    
    def _extract_path_thread(self):
        try:
            prune_threshold = self.prune_var.get()
            keep_count = self.keep_paths_var.get()
            
            # ===== 1. 骨骼化 =====
            binary_01 = (self.dilated_binary == 255).astype(np.uint8)
            skeleton_01 = morph.skeletonize(binary_01)
            skeleton_255 = (skeleton_01 * 255).astype(np.uint8)
            
            # ===== 2. 去毛刺 =====
            skeleton_pruned = self._prune_short_branches(skeleton_255, prune_threshold)
            self.skeleton = skeleton_pruned
            
            # ===== 3. 提取所有路径 =====
            all_paths = self._extract_all_paths(skeleton_pruned)
            
            # ===== 4. 去分叉 =====
            main_paths = self._remove_branches(all_paths)
            
            # ===== 5. 按长度排序，保留前N条 =====
            if len(main_paths) > 0:
                main_paths.sort(key=len, reverse=True)
                self.selected_paths = main_paths[:keep_count]
            else:
                self.selected_paths = []
            
            # ===== 6. 生成路径颜色 =====
            self.path_colors = self._generate_colors(len(self.selected_paths))
            
            # ===== 7. 更新窗口3: 原图 + 彩色路径 =====
            path_img = self.img_color.copy()
            self._draw_paths_on_image(path_img, self.selected_paths, self.path_colors)
            
            self.root.after(0, self._update_window3, path_img)
            
            # 统计信息
            path_info = f"保留 {len(self.selected_paths)} 条路径"
            if len(self.selected_paths) > 0:
                path_info += f", 最长: {len(self.selected_paths[0])}点"
            
            self.root.after(0, self._show_status, f"✅ 路径提取完成: {path_info}")
            self.root.after(0, self._update_progress, path_info)
            self.root.after(0, self._update_info, path_info)
            
            self.btn_path.config(state="normal")
            
        except Exception as e:
            self.root.after(0, self._show_status, f"❌ 路径提取出错: {str(e)}")
            self.root.after(0, self._enable_path_btn)
            import traceback
            traceback.print_exc()
    
    def _find_segments(self, pixel_array, threshold):
        """查找一维数组中所有灰度值 < 阈值的连续段"""
        segments = []
        in_segment = False
        start = 0
        
        for i, val in enumerate(pixel_array):
            if val < threshold and not in_segment:
                in_segment = True
                start = i
            elif (val >= threshold or i == len(pixel_array) - 1) and in_segment:
                if i == len(pixel_array) - 1 and val < threshold:
                    end = i
                else:
                    end = i - 1
                if end >= start:
                    segments.append((start, end))
                in_segment = False
        return segments
    
    def _find_endpoints(self, skeleton_bool):
        """查找骨骼的端点"""
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
        """从端点追踪分支"""
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
        """去毛刺：删除短分支"""
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
    
    def _extract_all_paths(self, skeleton_255):
        """从骨骼图像中提取所有路径"""
        h, w = skeleton_255.shape
        visited = np.zeros((h, w), dtype=bool)
        paths = []
        
        skeleton_bool = skeleton_255 == 255
        skeleton_points = np.argwhere(skeleton_bool)
        
        if len(skeleton_points) == 0:
            return paths
        
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for y, x in skeleton_points:
            if visited[y, x]:
                continue
            
            queue = deque([(y, x)])
            visited[y, x] = True
            path_points = [(x, y)]
            
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in directions:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if not visited[ny, nx] and skeleton_bool[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                            path_points.append((nx, ny))
            
            if len(path_points) >= 10:
                paths.append(self._sort_path_points(path_points))
        
        return paths
    
    def _sort_path_points(self, points):
        """对路径点进行排序，使其形成连续路径"""
        if len(points) <= 2:
            return points
        
        sorted_points = [points[0]]
        remaining = points[1:]
        
        while remaining:
            last = sorted_points[-1]
            min_dist = float('inf')
            min_idx = 0
            for i, p in enumerate(remaining):
                dist = (last[0] - p[0])**2 + (last[1] - p[1])**2
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            sorted_points.append(remaining.pop(min_idx))
        
        return sorted_points
    
    def _remove_branches(self, all_paths):
        """去分叉：合并连接的分支"""
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
        """判断两条路径是否相连"""
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
        """生成不同的颜色"""
        colors = [
            (0, 0, 255),      # 红色
            (0, 165, 255),    # 橙色
            (0, 255, 255),    # 黄色
            (255, 0, 255),    # 品红
            (255, 255, 0),    # 青色
            (128, 0, 255),    # 紫色
            (0, 200, 0),      # 绿色
            (255, 128, 0),    # 蓝绿色
            (200, 200, 0),    # 黄绿色
            (0, 0, 200),      # 深红
        ]
        
        result = []
        for i in range(count):
            result.append(colors[i % len(colors)])
        return result
    
    def _draw_paths_on_image(self, img, paths, colors):
        """在图像上绘制路径"""
        for idx, path in enumerate(paths):
            if len(path) < 2:
                continue
            
            color = colors[idx % len(colors)]
            
            # 绘制路径线条
            for i in range(len(path) - 1):
                cv2.line(img, path[i], path[i+1], color, 2)
            
            # 起点：黄色空心圆
            cv2.circle(img, path[0], 6, (0, 255, 255), 2)
            
            # 终点：黄色实心圆
            cv2.circle(img, path[-1], 6, (0, 255, 255), -1)
            
            # 在路径旁边标注编号
            mid_idx = len(path) // 2
            cv2.putText(img, f"#{idx+1}", (path[mid_idx][0] + 5, path[mid_idx][1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _update_window1(self, img):
        """更新窗口1：原图 + 标记"""
        if self.zoom_orig is not None:
            self.zoom_orig.update_image(img)
    
    def _update_window2(self, img):
        """更新窗口2：二值图 / 膨胀图"""
        if self.zoom_binary is not None:
            self.zoom_binary.update_image(img)
    
    def _update_window3(self, img):
        """更新窗口3：原图 + 路径"""
        if self.zoom_path is not None:
            self.zoom_path.update_image(img)
    
    def _update_progress(self, text):
        self.progress_label.config(text=text)
    
    def _update_info(self, text):
        self.info_label.config(text=f"路径信息: {text}")
    
    def _enable_scan_btn(self):
        self.btn_scan.config(state="normal")
    
    def _enable_dilate_btn(self):
        self.btn_dilate.config(state="normal")
    
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
    