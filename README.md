# Whale1.0 — 开源视频生成大模型

> 🐋 整合全球顶尖开源视频生成模型，打造统一的视频生成平台

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## 概述

**Whale1.0** 是一个集成了当前最先进开源视频生成模型的统一平台，涵盖文本到视频、图像到视频、人像动画、口型同步、帧插值等完整视频生成能力链。

### 技术架构

```
Whale1.0
├── 核心引擎 (Engine Layer)
│   ├── Wan2.2      — MoE 架构，电影级视频生成（阿里巴巴）
│   ├── CogVideoX   — DiT 架构，中文强语义理解（智谱AI/清华）
│   ├── HunyuanVideo— 轻量级 DiT，消费级显卡可跑（腾讯）
│   ├── Mochi       — 高质量文生视频（Genmo）
│   └── LTX-Video   — 22B DiT，音画同步（Lightricks）
├── 增强模块 (Module Layer)
│   ├── LivePortrait — 人像动画驱动
│   ├── Wav2Lip      — 视频口型同步
│   ├── AnimateDiff  — Stable Diffusion 动画化
│   └── RIFE         — 实时帧插值
└── 综合管线 (Pipeline Layer)
    ├── T2V Pipeline       — 文本到视频
    ├── I2V Pipeline       — 图片到视频
    ├── Portrait Pipeline  — 人像视频生成
    └── Full Pipeline      — 全链路视频生产
```

### 核心特性

- 🎯 **统一接口** — 所有引擎使用相同的调用接口，一行代码切换模型
- 🔌 **模块化设计** — 核心引擎、增强模块、综合管线可自由组合
- 🚀 **智能调度** — 根据输入自动选择最优模型或级联调用
- 🎬 **全链路生产** — 从文本/图像到最终成片，完整的视频生产管线
- 🐳 **Docker 支持** — 一键部署，环境零配置

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/seanlab007/Whale1.0.git
cd Whale1.0

# 初始化子模块（所有 fork 仓库）
bash scripts/init_submodules.sh

# 安装依赖
pip install -r requirements/base.txt

# 下载模型权重
bash scripts/download_models.sh

# 生成视频
python scripts/run_pipeline.py --prompt "一只在雪地里奔跑的北极熊" --engine wan2.2
```

## 子模块

Whale1.0 集成了 11 个顶级开源仓库，全部 fork 至 `github.com/seanlab007/`：

```bash
# 初始化所有子模块（浅克隆，建议首选）
bash scripts/init_submodules.sh

# 完整克隆（获取完整 git 历史）
bash scripts/init_submodules.sh --depth 0

# 初始化单个
git clone --depth 1 https://github.com/seanlab007/Wan2.2.git submodules/Wan2.2
```

## 四端同步

项目同步到以下四个平台：

| 平台 | URL | 状态 |
|------|-----|------|
| GitHub | https://github.com/seanlab007/Whale1.0 | ✅ |
| GitLab | https://gitlab.com/seanlab007/Whale1.0 | ✅ |
| Gitee | https://gitee.com/seanlab007/Whale1.0 | ✅ |
| 本地 | `/path/to/whale1.0` | ✅ |

## 模型注册表

| 模型 | 类型 | 参数 | 架构 | 显存要求 | 许可证 |
|------|------|------|------|----------|--------|
| Wan2.2-T2V | 文生视频 | 14B | MoE | 24GB+ | Apache 2.0 |
| Wan2.2-I2V | 图生视频 | 14B | MoE | 24GB+ | Apache 2.0 |
| CogVideoX-2 | 文生视频 | 13B | DiT | 16GB+ | Apache 2.0 |
| HunyuanVideo 1.5 | 文生视频 | 8.3B | DiT | 12GB+ | Apache 2.0 |
| Mochi-1 | 文生视频 | 10B | DiT | 20GB+ | Apache 2.0 |
| LTX-Video 2.3 | 文+音视频 | 22B | DiT | 16GB+ | Apache 2.0 |
| SkyReels-V1 | 人物视频 | - | DiT | 16GB+ | Apache 2.0 |
| LivePortrait | 人像动画 | - | - | 8GB+ | MIT |
| Wav2Lip | 口型同步 | - | - | 4GB+ | Apache 2.0 |
| AnimateDiff | 动画化 | - | UNet | 8GB+ | Apache 2.0 |
| RIFE | 帧插值 | - | - | 4GB+ | MIT |

## 许可证

本项目基于 Apache 2.0 许可证开源，集成的子模型遵循其各自的许可证。

## 致谢

- [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [THUDM/CogVideo](https://github.com/THUDM/CogVideo)
- [Tencent-Hunyuan/HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo)
- [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora)
- [genmoai/mochi](https://github.com/genmoai/mochi)
- [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)
- [SkyworkAI/SkyReels-V1](https://github.com/SkyworkAI/SkyReels-V1)
- [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait)
- [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- [guoyww/AnimateDiff](https://github.com/guoyww/AnimateDiff)
- [megvii-research/ECCV2022-RIFE](https://github.com/megvii-research/ECCV2022-RIFE)
