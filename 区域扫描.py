"""
区域像素填充工具 - 实时预览版
- 扫描原图，找到灰度值 < 暗色阈值 的像素
- 以该像素为左上角，取 窗口大小 x 窗口大小 的区域
- 如果该区域内 >= 比例阈值 的像素灰度值 < 暗色阈值，则将该区域所有像素按原图位置填充到白色新图
- 滑块可调：窗口大小、比例阈值、暗色阈值、扫描步长
- 新增：实时显示扫描框位置和填充过程（逐行扫描动画）
- 新增：一键生成功能（只生成不保存）
- 新增：鼠标滚轮缩放（以鼠标为中心），双击显示全图
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
import threading
import time
import math

class ZoomableCanvas:
    """可缩放画布类，管理单个画布的缩放和显示"""
    def __init__(self, canvas, image_cv):
        self.canvas = canvas
        self.image_cv = image_cv  # BGR格式
        self.image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(self.image_rgb)
        
        self.zoom_level = 1.0
        self.min_zoom = 0.05
        self.max_zoom = 20.0
        self.offset_x = 0
        self.offset_y = 0
        
        # 当前显示的PhotoImage
        self.photo_image = None
        
        # 绑定鼠标事件
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        # 绑定进入事件，确保鼠标在画布上时能接收到滚轮事件
        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)
        
        # 拖拽相关
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.is_dragging = False
        self.mouse_inside = False
        
        self.update_display()
    
    def _on_enter(self, event):
        """鼠标进入画布"""
        self.mouse_inside = True
        # 让画布获得焦点以接收滚轮事件
        self.canvas.focus_set()
    
    def _on_leave(self, event):
        """鼠标离开画布"""
        self.mouse_inside = False
    
    def update_image(self, image_cv):
        """更新显示的图片"""
        self.image_cv = image_cv
        self.image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(self.image_rgb)
        # 重置缩放和偏移
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update_display()
    
    def update_display(self):
        """更新画布显示"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            return
        
        # 计算缩放后的图像大小
        pw, ph = self.pil_image.size
        display_w = pw * self.zoom_level
        display_h = ph * self.zoom_level
        
        # 如果缩放级别使图像小于画布，居中显示
        if display_w < canvas_width and display_h < canvas_height:
            # 保持当前缩放，居中显示
            offset_x = (canvas_width - display_w) / 2
            offset_y = (canvas_height - display_h) / 2
        else:
            # 应用用户拖拽偏移
            offset_x = self.offset_x
            offset_y = self.offset_y
            
            # 限制偏移范围，防止图像完全移出视野
            max_off_x = max(0, (display_w - canvas_width) / 2)
            max_off_y = max(0, (display_h - canvas_height) / 2)
            offset_x = max(-max_off_x, min(max_off_x, offset_x))
            offset_y = max(-max_off_y, min(max_off_y, offset_y))
            self.offset_x = offset_x
            self.offset_y = offset_y
        
        # 裁剪和缩放图像
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
            canvas_width//2 + offset_x, 
            canvas_height//2 + offset_y, 
            image=self.photo_image, 
            anchor=tk.CENTER
        )
    
    def _on_mousewheel(self, event):
        """鼠标滚轮缩放 - 以鼠标位置为中心"""
        if not self.mouse_inside:
            return
        
        # 获取鼠标在画布上的位置
        x = event.x
        y = event.y
        
        # 计算缩放因子
        if event.delta > 0:
            factor = 1.1
        else:
            factor = 0.9
        
        # 计算新的缩放级别
        new_zoom = self.zoom_level * factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        if new_zoom == self.zoom_level:
            return
        
        # 获取画布尺寸
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        pw, ph = self.pil_image.size
        
        # 计算当前显示尺寸
        current_w = pw * self.zoom_level
        current_h = ph * self.zoom_level
        
        # 计算鼠标在图像坐标系中的位置（相对于图像左上角，单位：像素）
        # 考虑当前的偏移和居中
        img_x = (x - canvas_width/2 - self.offset_x) / current_w * pw
        img_y = (y - canvas_height/2 - self.offset_y) / current_h * ph
        
        # 应用缩放
        self.zoom_level = new_zoom
        
        # 计算新的偏移，使鼠标指向的图像位置保持不变
        new_w = pw * self.zoom_level
        new_h = ph * self.zoom_level
        
        # 新偏移 = 鼠标位置 - 图像在画布中的新位置
        # 图像在画布中的位置 = 鼠标在图像中的位置 * 缩放比例 + 偏移
        self.offset_x = x - canvas_width/2 - (img_x / pw) * new_w
        self.offset_y = y - canvas_height/2 - (img_y / ph) * new_h
        
        self.update_display()
    
    def _on_mouse_down(self, event):
        """鼠标按下，准备拖拽"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.drag_offset_x = self.offset_x
        self.drag_offset_y = self.offset_y
        self.is_dragging = True
    
    def _on_mouse_drag(self, event):
        """鼠标拖拽平移"""
        if not self.is_dragging:
            return
        
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        self.offset_x = self.drag_offset_x + dx
        self.offset_y = self.drag_offset_y + dy
        
        self.update_display()
    
    def _on_double_click(self, event):
        """双击显示全图（适应画布大小）"""
        # 获取画布尺寸
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        pw, ph = self.pil_image.size
        
        if canvas_width < 10 or canvas_height < 10:
            return
        
        # 计算适应画布的缩放比例
        fit_scale_x = canvas_width / pw
        fit_scale_y = canvas_height / ph
        fit_scale = min(fit_scale_x, fit_scale_y)
        
        # 应用适应缩放
        self.zoom_level = fit_scale
        self.offset_x = 0
        self.offset_y = 0
        
        self.update_display()
    
    def get_display_transform(self):
        """获取显示变换信息，用于绘制扫描框"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        pw, ph = self.pil_image.size
        
        # 图像在画布中的显示尺寸
        display_w = pw * self.zoom_level
        display_h = ph * self.zoom_level
        
        # 图像在画布中的位置（左上角）
        img_left = canvas_width/2 + self.offset_x - display_w/2
        img_top = canvas_height/2 + self.offset_y - display_h/2
        
        return {
            'img_left': img_left,
            'img_top': img_top,
            'display_w': display_w,
            'display_h': display_h,
            'scale_x': display_w / pw,
            'scale_y': display_h / ph
        }


class RegionFillTool:
    def __init__(self, root):
        self.root = root
        self.root.title("区域像素填充工具 - 实时预览版")
        self.root.geometry("1300x750")

        self.img_path = None
        self.img_color = None      # 原图 BGR
        self.img_gray = None       # 原图灰度
        self.result = None         # 当前结果图
        
        # 实时预览相关
        self.is_running = False
        self.pause_preview = False
        self.preview_speed = 0.5  # 扫描速度（秒/行）
        self.current_line = 0      # 当前扫描到的行
        self.current_col = 0       # 当前扫描到的列
        self.scan_complete = False
        
        # 缩放画布对象
        self.orig_zoom = None
        self.result_zoom = None

        self._build_ui()
        self._init_overlay_images()
        
        # 添加鼠标提示
        self._show_mouse_tips()

    def _show_mouse_tips(self):
        """显示鼠标操作提示"""
        tip_text = "💡 滚轮缩放（以鼠标为中心）| 拖拽平移 | 双击显示全图"
        self.status.config(text=tip_text)

    def _init_overlay_images(self):
        """初始化覆盖层图像缓存"""
        self._overlay_orig = None
        self._overlay_result = None

    def _build_ui(self):
        # 顶部控制区
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        # 第一行：文件选择和参数
        param_frame = ttk.Frame(ctrl)
        param_frame.pack(side=tk.TOP, fill=tk.X)

        # 文件选择
        ttk.Button(param_frame, text="选择原图", command=self._load_image).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(param_frame, text="未选择文件")
        self.file_label.pack(side=tk.LEFT, padx=5)

        ttk.Separator(param_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # 窗口大小滑块 (3 ~ 100)
        ttk.Label(param_frame, text="窗口:").pack(side=tk.LEFT, padx=(10, 2))
        self.win_var = tk.IntVar(value=16)
        self.win_label = ttk.Label(param_frame, text="16", width=4)
        self.win_label.pack(side=tk.LEFT)
        self.win_scale = tk.Scale(param_frame, from_=3, to=100, variable=self.win_var,
                                    orient=tk.HORIZONTAL, command=self._on_win_change)
        self.win_scale.pack(side=tk.LEFT, padx=(2, 10))

        # 比例阈值滑块 (10% ~ 100%)
        ttk.Label(param_frame, text="比例:").pack(side=tk.LEFT, padx=(10, 2))
        self.ratio_var = tk.DoubleVar(value=90.0)
        self.ratio_label = ttk.Label(param_frame, text="90%", width=5)
        self.ratio_label.pack(side=tk.LEFT)
        self.ratio_scale = tk.Scale(param_frame, from_=10, to=100, variable=self.ratio_var,
                                      orient=tk.HORIZONTAL, command=self._on_ratio_change)
        self.ratio_scale.pack(side=tk.LEFT, padx=(2, 10))

        # 暗色阈值滑块 (1 ~ 255)
        ttk.Label(param_frame, text="暗色:").pack(side=tk.LEFT, padx=(10, 2))
        self.dark_var = tk.IntVar(value=180)
        self.dark_label = ttk.Label(param_frame, text="180", width=4)
        self.dark_label.pack(side=tk.LEFT)
        self.dark_scale = tk.Scale(param_frame, from_=1, to=255, variable=self.dark_var,
                                     orient=tk.HORIZONTAL, command=self._on_dark_change)
        self.dark_scale.pack(side=tk.LEFT, padx=(2, 10))

        # 扫描步长滑块 (1 ~ 20)
        ttk.Label(param_frame, text="步长:").pack(side=tk.LEFT, padx=(10, 2))
        self.step_var = tk.IntVar(value=1)
        self.step_label = ttk.Label(param_frame, text="1", width=3)
        self.step_label.pack(side=tk.LEFT)
        self.step_scale = tk.Scale(param_frame, from_=1, to=20, variable=self.step_var,
                                    orient=tk.HORIZONTAL, command=self._on_step_change)
        self.step_scale.pack(side=tk.LEFT, padx=(2, 10))

        # 第二行：操作按钮
        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))

        # 一键生成按钮（放在最前面，最醒目）
        self.gen_btn = ttk.Button(btn_frame, text="🚀 一键生成", command=self._one_click_generate)
        self.gen_btn.pack(side=tk.LEFT, padx=5)
        # 设置按钮样式
        style = ttk.Style()
        style.configure("Generate.TButton", foreground="green", font=("Arial", 10, "bold"))
        self.gen_btn.config(style="Generate.TButton")

        ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # 实时预览控制
        ttk.Button(btn_frame, text="▶ 实时预览", command=self._start_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="⏸ 暂停", command=self._toggle_pause).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="⏹ 停止", command=self._stop_preview).pack(side=tk.LEFT, padx=5)

        ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # 速度滑块
        ttk.Label(btn_frame, text="速度:").pack(side=tk.LEFT, padx=(10, 2))
        self.speed_var = tk.DoubleVar(value=0.5)
        self.speed_label = ttk.Label(btn_frame, text="0.5s", width=5)
        self.speed_label.pack(side=tk.LEFT)
        self.speed_scale = tk.Scale(btn_frame, from_=0.05, to=2.0, variable=self.speed_var,
                                      orient=tk.HORIZONTAL, command=self._on_speed_change)
        self.speed_scale.pack(side=tk.LEFT, padx=(2, 10))

        # 快速执行按钮
        ttk.Button(btn_frame, text="⚡ 快速填充", command=self._run_fill).pack(side=tk.LEFT, padx=5)

        # 进度显示
        self.progress_label = ttk.Label(btn_frame, text="", font=("Arial", 9))
        self.progress_label.pack(side=tk.LEFT, padx=10)

        # 图片显示区 - 两栏：原图 | 新图
        canvas_frame = ttk.Frame(self.root, padding=5)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 左侧原图（带覆盖层，显示扫描框）
        left = ttk.LabelFrame(canvas_frame, text="原图 (滚轮缩放 | 拖拽平移 | 双击全图)")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.canvas_orig = tk.Canvas(left, bg="gray", highlightthickness=1, highlightcolor="red")
        self.canvas_orig.pack(fill=tk.BOTH, expand=True)
        # 绑定画布大小变化事件
        self.canvas_orig.bind("<Configure>", self._on_canvas_resize)

        # 右侧新图（填充结果）
        right = ttk.LabelFrame(canvas_frame, text="新图（填充结果） (滚轮缩放 | 拖拽平移 | 双击全图)")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.canvas_result = tk.Canvas(right, bg="gray")
        self.canvas_result.pack(fill=tk.BOTH, expand=True)

        # 底部状态
        self.status = ttk.Label(self.root, text="💡 滚轮缩放（以鼠标为中心）| 拖拽平移 | 双击显示全图", relief=tk.SUNKEN)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self._orig_tk = None
        self._result_tk = None
        
        # 扫描框相关
        self.scan_rect_id = None
        self.orig_display_size = None

    def _on_canvas_resize(self, event):
        """画布大小变化时重新绘制"""
        # 更新缩放画布显示
        if self.orig_zoom is not None:
            self.orig_zoom.update_display()
        if self.result_zoom is not None:
            self.result_zoom.update_display()
        
        if self.scan_rect_id is not None and self.is_running:
            self._update_scan_box()

    def _on_win_change(self, val):
        v = int(float(val))
        self.win_var.set(v)
        self.win_label.config(text=str(v))

    def _on_ratio_change(self, val):
        v = round(float(val), 1)
        self.ratio_var.set(v)
        self.ratio_label.config(text=f"{v}%")

    def _on_dark_change(self, val):
        v = int(float(val))
        self.dark_var.set(v)
        self.dark_label.config(text=str(v))

    def _on_step_change(self, val):
        v = int(float(val))
        self.step_var.set(v)
        self.step_label.config(text=str(v))

    def _on_speed_change(self, val):
        v = round(float(val), 2)
        self.speed_var.set(v)
        self.speed_label.config(text=f"{v}s")
        self.preview_speed = v

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="选择原图",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("所有文件", "*.*")]
        )
        if not path:
            return

        self.img_color = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.img_color is None:
            self.status.config(text=f"无法读取图片: {path}")
            return

        self.img_gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        self.img_path = path
        self.file_label.config(text=os.path.basename(path))
        self.status.config(text=f"已加载: {path}  尺寸: {self.img_color.shape[1]}x{self.img_color.shape[0]}")

        # 创建可缩放画布
        self.orig_zoom = ZoomableCanvas(self.canvas_orig, self.img_color)
        self.result = np.ones(self.img_color.shape, dtype=np.uint8) * 255
        self.result_zoom = ZoomableCanvas(self.canvas_result, self.result)
        
        self.scan_rect_id = None
        self.is_running = False
        self.scan_complete = False
        self.progress_label.config(text="")
        self._show_mouse_tips()

    def _update_scan_box(self):
        """更新原图上的扫描框位置（逐行扫描动画）"""
        if self.scan_rect_id is not None:
            self.canvas_orig.delete(self.scan_rect_id)
            self.scan_rect_id = None
        
        if not self.is_running or self.img_gray is None or self.orig_zoom is None:
            return
        
        h, w = self.img_gray.shape
        step = self.step_var.get()
        line = self.current_line
        col = self.current_col
        
        # 如果扫描完成，不显示扫描框
        if line >= h or self.scan_complete:
            return
        
        # 获取显示变换信息
        transform = self.orig_zoom.get_display_transform()
        img_left = transform['img_left']
        img_top = transform['img_top']
        scale_x = transform['scale_x']
        scale_y = transform['scale_y']
        
        # 计算扫描框在显示坐标中的位置
        # 框的宽度 = 步长，高度 = 步长（表示当前扫描步进的跨度）
        x1 = img_left + col * scale_x
        y1 = img_top + line * scale_y
        x2 = img_left + (col + step) * scale_x
        y2 = img_top + (line + step) * scale_y
        
        # 确保框在图像范围内
        if col + step > w:
            x2 = img_left + w * scale_x
        if line + step > h:
            y2 = img_top + h * scale_y
        
        # 绘制红色矩形框（醒目边框 + 半透明填充）
        self.scan_rect_id = self.canvas_orig.create_rectangle(
            x1, y1, x2, y2,
            outline="red", width=3, tags="scan_box"
        )
        # 添加半透明填充，让扫描框更醒目
        self.canvas_orig.create_rectangle(
            x1, y1, x2, y2,
            fill="red", stipple="gray50", outline="", tags="scan_box"
        )
        
        # 更新进度信息
        total_lines = (h - self.win_var.get()) // step + 1 if h >= self.win_var.get() else 0
        current_line_idx = line // step + 1
        progress = min(100, int((line + step) / h * 100))
        self.progress_label.config(
            text=f"扫描: 行 {current_line_idx}/{total_lines}  ({progress}%) | 步长:{step} | 列:{col}"
        )

    def _start_preview(self):
        """启动实时预览 - 逐行扫描"""
        if self.img_gray is None:
            self.status.config(text="请先选择原图")
            return
        
        if self.is_running:
            return
        
        self.is_running = True
        self.pause_preview = False
        self.scan_complete = False
        self.current_line = 0
        self.current_col = 0
        
        # 重置结果图为全白
        h, w = self.img_gray.shape
        self.result = np.ones((h, w, 3), dtype=np.uint8) * 255
        if self.result_zoom is not None:
            self.result_zoom.update_image(self.result)
        
        self.status.config(text="🔄 实时预览进行中...")
        
        # 在新线程中运行扫描
        thread = threading.Thread(target=self._preview_scan, daemon=True)
        thread.start()

    def _preview_scan(self):
        """实时扫描线程 - 逐行扫描动画"""
        h, w = self.img_gray.shape
        s = self.win_var.get()
        step = self.step_var.get()
        ratio_threshold = self.ratio_var.get() / 100.0
        dark_threshold = self.dark_var.get()
        
        # 预计算二值化
        binary = (self.img_gray < dark_threshold).astype(np.float32)
        
        # 积分图
        pad = np.zeros((h + 1, w + 1), dtype=np.float64)
        pad[1:, 1:] = np.cumsum(np.cumsum(binary, axis=0), axis=1)
        
        # 结果图
        result = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # 逐行扫描（步长控制）
        for y in range(0, h - s + 1, step):
            if not self.is_running:
                break
            while self.pause_preview:
                time.sleep(0.1)
                if not self.is_running:
                    return
            
            self.current_line = y
            
            # 逐列扫描这一行
            for x in range(0, w - s + 1, step):
                if not self.is_running:
                    break
                while self.pause_preview:
                    time.sleep(0.1)
                    if not self.is_running:
                        return
                
                self.current_col = x
                
                # 检查起点是否暗像素
                if binary[y, x] == 1:
                    # 计算窗口内暗像素比例
                    count = (pad[y+s, x+s] - pad[y, x+s] - pad[y+s, x] + pad[y, x])
                    ratio = count / (s * s)
                    
                    if ratio >= ratio_threshold:
                        # 填充整个窗口
                        result[y:y+s, x:x+s] = self.img_color[y:y+s, x:x+s]
                
                # 每扫描一个窗口就更新显示（实时反馈）
                self.root.after(0, self._update_preview_display, result.copy())
                self.root.after(0, self._update_scan_box)
                
                # 速度控制（微调，让动画流畅）
                time.sleep(self.preview_speed * 0.01)
            
            # 行扫描完成，更新显示
            self.root.after(0, self._update_preview_display, result.copy())
            self.root.after(0, self._update_scan_box)
        
        # 扫描完成
        self.scan_complete = True
        self.result = result
        self.root.after(0, self._on_preview_complete)

    def _update_preview_display(self, result):
        """更新预览显示"""
        if result is not None and self.result_zoom is not None:
            self.result = result
            self.result_zoom.update_image(result)

    def _on_preview_complete(self):
        """预览完成回调"""
        self.is_running = False
        if self.scan_rect_id is not None:
            self.canvas_orig.delete(self.scan_rect_id)
            self.scan_rect_id = None
        self.status.config(text="✅ 实时预览完成")
        self.progress_label.config(text="扫描完成 100%")

    def _toggle_pause(self):
        """暂停/继续切换"""
        if not self.is_running:
            return
        self.pause_preview = not self.pause_preview
        if self.pause_preview:
            self.status.config(text="⏸ 已暂停")
        else:
            self.status.config(text="▶ 继续扫描...")

    def _stop_preview(self):
        """停止预览"""
        self.is_running = False
        self.pause_preview = False
        self.scan_complete = True
        if self.scan_rect_id is not None:
            self.canvas_orig.delete(self.scan_rect_id)
            self.scan_rect_id = None
        self.status.config(text="⏹ 已停止")
        self.progress_label.config(text="")

    def _run_fill(self):
        """快速填充（不显示扫描过程）- 支持步长控制"""
        if self.img_gray is None:
            self.status.config(text="请先选择原图")
            return

        win_size = self.win_var.get()
        step = self.step_var.get()
        ratio_threshold = self.ratio_var.get() / 100.0
        dark_threshold = self.dark_var.get()

        self.status.config(text=f"⚡ 正在处理... 窗口={win_size}, 步长={step}, 比例={ratio_threshold*100:.0f}%, 暗色阈值={dark_threshold}")
        self.root.update()

        h, w = self.img_gray.shape
        s = win_size

        # Step 1: 二值化
        binary = (self.img_gray < dark_threshold).astype(np.float32)

        # Step 2: 积分图
        pad = np.zeros((h + 1, w + 1), dtype=np.float64)
        pad[1:, 1:] = np.cumsum(np.cumsum(binary, axis=0), axis=1)

        if h >= s and w >= s:
            # 使用步长创建采样网格
            y_indices = np.arange(0, h - s + 1, step)
            x_indices = np.arange(0, w - s + 1, step)
            
            # 创建网格坐标
            y_grid, x_grid = np.meshgrid(y_indices, x_indices, indexing='ij')
            
            # 计算每个采样点的窗口暗像素比例
            count = (pad[y_grid + s, x_grid + s] - pad[y_grid, x_grid + s] - 
                     pad[y_grid + s, x_grid] + pad[y_grid, x_grid])
            ratio_map = count / (s * s)
            
            # 检查起点是否暗像素
            is_dark = binary[y_grid, x_grid] == 1
            
            # 找到满足条件的起点
            valid_mask = is_dark & (ratio_map >= ratio_threshold)
            
            # 创建有效起点矩阵（全图大小）
            valid_starts = np.zeros((h, w), dtype=np.uint8)
            valid_starts[y_grid[valid_mask], x_grid[valid_mask]] = 1
        else:
            valid_starts = np.zeros((h, w), dtype=np.uint8)

        # Step 4: 膨胀生成掩码
        kernel = np.ones((s, s), dtype=np.uint8)
        fill_mask = cv2.dilate(valid_starts, kernel, anchor=(0, 0))

        # Step 5: 填充
        result = np.ones((h, w, 3), dtype=np.uint8) * 255
        mask3 = np.stack([fill_mask] * 3, axis=-1) > 0
        result[mask3] = self.img_color[mask3]

        self.result = result
        if self.result_zoom is not None:
            self.result_zoom.update_image(result)

        filled = int(np.sum(fill_mask > 0))
        total = h * w
        
        # 计算实际检查的窗口数量
        num_y = (h - s) // step + 1 if h >= s else 0
        num_x = (w - s) // step + 1 if w >= s else 0
        total_checks = num_y * num_x
        
        self.status.config(text=f"✅ 完成! 已填充 {filled}/{total} 像素 ({filled/total*100:.1f}%) | 检查窗口: {total_checks}")

    def _one_click_generate(self):
        """一键生成功能"""
        if self.img_gray is None:
            self.status.config(text="❌ 请先选择原图")
            return
        
        # 如果有正在运行的预览，先停止
        if self.is_running:
            self._stop_preview()
            time.sleep(0.1)
        
        # 执行快速填充
        self._run_fill()
        
        # 显示完成信息
        if self.result is not None:
            h, w = self.result.shape[:2]
            white_pixels = np.sum(np.all(self.result == [255, 255, 255], axis=-1))
            total_pixels = h * w
            fill_ratio = (1 - white_pixels / total_pixels) * 100
            self.status.config(text=f"✅ 一键生成完成！尺寸: {w}x{h}  填充: {fill_ratio:.1f}%")
            self.progress_label.config(text="")
        else:
            self.status.config(text="❌ 一键生成失败")


if __name__ == "__main__":
    root = tk.Tk()
    app = RegionFillTool(root)
    root.mainloop()