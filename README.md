!密封胶路路径查找
<img width="1079" height="538" alt="reame" src="https://github.com/lister2000/Exploration-path-in-the-diagram/blob/master/images/reame.png" />


# 🔍 SearchPath - 路径搜索算法库

[![GitHub stars](https://img.shields.io/github/stars/lister2000/myacademy?style=social)](https://github.com/lister2000/myacademy/stargazers)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/lister2000/myacademy)](https://github.com/lister2000/myacademy/commits/master)

> 🚀 一个高性能的路径搜索算法库，包含三种核心算法实现

---

## 📖 目录导航

- [✨ 项目简介](#-项目简介)
- [🧩 算法导航](#-算法导航)
---

## ✨ 项目简介

**SearchPath** 是关于路径搜索的算法探索，提供了三种不同的算法实现：

---

## 🧩 算法导航

### 1. 区域扫描算法 (Region Scanning)

> 📍 **算法简介**：通过划分搜索区域，逐块扫描寻找最优路径。

#### 核心原理
```
─────────────────────────────────────────────────────────────────
                      区域扫描算法流程                          
─────────────────────────────────────────────────────────────────
                                                                
  输入: 灰度图、灰度阈值、窗口高度(H)、窗口宽度(W)                   
                                                                
  步骤1: 在图像上滑动矩形窗口 (H × W)                            
         从左上角开始，步长为1像素                                 
                                                             
  步骤2: 计算窗口内所有像素的平均灰度值                            
         avg = sum(窗口内像素) / (H × W)                         
                                                               
  步骤3: 判断                                                   
         if avg <= 灰度阈值:                                    
             窗口内所有像素标记为白色(255)  → 小路                
         else:                                                 
             窗口内所有像素保持黑色(0)     → 背景                  
                                                                  
  步骤4: 窗口继续滑动，直到遍历整个图像                             
                                                                 
  输出: 二值图 (黑色背景，白色标记)                                 
                                                                  
─────────────────────────────────────────────────────────────────

─────────────────────────────────────────────────────────────────
                      窗口内判断                                  
─────────────────────────────────────────────────────────────────
  ┌──┬──┬──┬──┬──┐                                         
  │65│68│70│72│75│  窗口内像素值                            
  ├──┼──┼──┼──┼──┤                                         
  │68│70│72│75│78│  平均值 = 75                            
  ├──┼──┼──┼──┼──┤  阈值 = 100                             
  │70│72│75│78│80│  75 <= 100 ✅                            
  ├──┼──┼──┼──┼──┤  标记整个窗口为白色                     
  │72│75│78│80│82│                                         
  ├──┼──┼──┼──┼──┤                                         
  │75│78│80│82│85│                                         
  └──┴──┴──┴──┴──┘                                         
                                                             
  ┌──┬──┬──┬──┬──┐                                         
  │150│148│145│142│140│  窗口内像素值                       
  ├──┼──┼──┼──┼──┤                                         
  │148│145│142│140│138│  平均值 = 142                      
  ├──┼──┼──┼──┼──┤  阈值 = 100                             
  │145│142│140│138│135│  142 > 100 ❌                       
  ├──┼──┼──┼──┼──┤  不标记                                
  │142│140│138│135│132│                                     
  ├──┼──┼──┼──┼──┤                                         
  │140│138│135│132│130│                                     
  └──┴──┴──┴──┴──┘                                         
─────────────────────────────────────────────────────────────
```
[⬆ 返回顶部](#-目录导航)

### 2. 路径提取算法 (Fath Finding)

```
─────────────────────────────────────────────────────────────────
                      提取路径算法流程                          
─────────────────────────────────────────────────────────────────
                                                                 
  二值图 (黑白)                                                  
       ↓                                                        
  步骤1: 骨骼化 (Skeletonization)                               
       ↓                                                        
  步骤2: 去毛刺 (Prune Short Branches)                          
       ↓                                                        
  步骤3: 提取连通路径 (Connected Components)                    
       ↓                                                        
  步骤4: Dijkstra 最长路径提取                                  
       ↓                                                        
  步骤5: 去分叉 (Remove Branches)                               
       ↓                                                        
  步骤6: 排序 & 保留前N条                                       
       ↓                                                        
  输出: 路径点列表 + 彩色路径图                                 
                                                                 
─────────────────────────────────────────────────────────────────

路径分叉:
┌─────────────────────────────────────────────────────────────┐
│                  ●                                          │
│                 /                                           │
│                /                                            │
│               /                                             │
│  ●───────────●───────────●───────────●───────────●          │
│              │                                              │
│              │                                              │
│              ●───────────●───────────●                      │ ← 分叉
│                                                             │
└─────────────────────────────────────────────────────────────┘
                    ↓ Dijkstra
最长路径:
┌─────────────────────────────────────────────────────────────┐
│  ●───────────●───────────●───────────●───────────●          │ ← 选择最长的分支
│              │                                              │
│              │                                              │
│              ●───────────●───────────●                      │ ← 分支被忽略
│                                                             │
└─────────────────────────────────────────────────────────────┘

```

## 📦 快速开始

### 安装
```bash
git clone https://github.com/Exploration-path-in-the-diagram.git
cd Exploration-path-in-the-diagram
```

### 使用示例

 python main.py

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">
  <sub>Made with ❤️ by <a href="https://github.com/lister2000">Lister</a></sub>
</div>


| 首页截图 | 详情页截图 | 设置页截图 |
| :---: | :---: | :---: |

<img width="1079" height="538" alt="reame" src="https://github.com/lister2000/Exploration-path-in-the-diagram/blob/master/images/%E7%95%8C%E9%9D%A2.png" />


# 🔍 SearchPath - 路径搜索算法库

[![GitHub stars](https://img.shields.io/github/stars/lister2000/myacademy?style=social)](https://github.com/lister2000/myacademy/stargazers)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/lister2000/myacademy)](https://github.com/lister2000/myacademy/commits/master)

> 🚀 一个高性能的路径搜索算法库，包含三种核心算法实现

---

## 📖 目录导航

- [✨ 项目简介](#-项目简介)
- [🧩 算法导航](#-算法导航)
  - [1. 区域扫描算法](#1-区域扫描算法-region-scanning)
  - [2. 切片扫描算法](#2-切片扫描算法-slice-scanning)
  - [3. 采样生成算法](#3-采样生成算法-sampling-generation)
- [📦 快速开始](#-快速开始)
- [📊 性能对比](#-性能对比)
- [🤝 贡献指南](#-贡献指南)
- [📄 许可证](#-许可证)

---

## ✨ 项目简介

**SearchPath** 是一个专注于路径搜索的算法库，提供了三种不同的算法实现，适用于不同的应用场景：

| 算法 | 适用场景 | 时间复杂度 |
|:---|:---|:---|
| 区域扫描 | 大范围地图、游戏寻路 | O(n²) |
| 切片扫描 | 分层数据、三维空间 | O(n·log n) |
| 采样生成 | 动态环境、实时路径 | O(n) |

---

## 🧩 算法导航

### 1. 区域扫描算法 (Region Scanning)

> 📍 **算法简介**：通过划分搜索区域，逐块扫描寻找最优路径。

#### 核心原理
```
┌─────────┬─────────┬─────────┐
│  区域A  │  区域B  │  区域C  │
├─────────┼─────────┼─────────┤
│  区域D  │  区域E  │  区域F  │
├─────────┼─────────┼─────────┤
│  区域G  │  区域H  │  区域I  │
└─────────┴─────────┴─────────┘
```
*图：区域扫描算法将地图划分为若干区域进行搜索*

#### 特点
- ✅ **优点**：适合大范围地图，易于并行处理
- ⚠️ **缺点**：区域划分策略影响效率
- 🎯 **最佳场景**：游戏寻路、城市规划

#### 代码示例
```python
# 区域扫描示例
def region_scanning(map_data, start, end):
    regions = divide_into_regions(map_data)
    return search_across_regions(regions, start, end)
```

[⬆ 返回顶部](#-目录导航)

---

### 2. 切片扫描算法 (Slice Scanning)

> 📍 **算法简介**：对数据在某个维度上进行切片，逐层扫描分析。

#### 核心原理
```
         Z轴切片
          ↑
    ┌─────┼─────┐
    │ 层1  │ 层2  │ 层3  │
    │      │      │      │
    └──────┴──────┴──────┘
          → X轴扫描
```
*图：切片扫描算法沿Z轴分层，在每层内进行X轴扫描*

#### 特点
- ✅ **优点**：适合分层数据，内存占用可控
- ⚠️ **缺点**：切片厚度影响精度
- 🎯 **最佳场景**：3D打印分层、医学影像分析

#### 代码示例
```python
# 切片扫描示例
def slice_scanning(volume_data, slice_thickness):
    slices = slice_data(volume_data, slice_thickness)
    return scan_slices(slices)
```

[⬆ 返回顶部](#-目录导航)

---

### 3. 采样生成算法 (Sampling Generation)

> 📍 **算法简介**：在搜索空间中随机采样，逐步优化生成路径。

#### 核心原理
```
      ⭐ 目标点
      ↑
    ┌──┼──┐
    │  │  │  ← 采样点分布
    │  ★  │
    │  │  │
    └──┼──┘
      ↓
      🏁 起始点
```
*图：采样生成算法在空间中随机采样，逐步收敛到最优路径*

#### 特点
- ✅ **优点**：适合高维空间，无需全局建模
- ⚠️ **缺点**：结果具有随机性
- 🎯 **最佳场景**：机器人运动规划、自动驾驶

#### 代码示例
```python
# 采样生成示例
def sampling_generation(space, sample_count):
    samples = random_sample(space, sample_count)
    return optimize_path(samples)
```

[⬆ 返回顶部](#-目录导航)

---

## 📦 快速开始

### 安装
```bash
git clone https://github.com/lister2000/myacademy.git
cd myacademy
```

### 使用示例
```python
from searchpath import RegionScan, SliceScan, SampleScan

# 区域扫描
result = RegionScan.scan(map_data, start, end)

# 切片扫描
result = SliceScan.scan(volume_data, slice_thickness=2)

# 采样生成
result = SampleScan.generate(space, sample_count=1000)
```

[⬆ 返回顶部](#-目录导航)

---

## 📊 性能对比

| 指标 | 区域扫描 | 切片扫描 | 采样生成 |
|:---|:---:|:---:|:---:|
| 时间复杂度 | O(n²) | O(n·log n) | O(n) |
| 内存占用 | 高 | 中 | 低 |
| 精度 | 高 | 中 | 中 |
| 适用维度 | 2D | 2D/3D | nD |

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">
  <sub>Made with ❤️ by <a href="https://github.com/lister2000">Lister</a></sub>
</div>




