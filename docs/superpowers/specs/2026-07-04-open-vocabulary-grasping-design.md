# Open-Vocabulary Robot Grasping — 项目设计文档

> 目标：打造一个 GitHub 数千 Star 级别的开源项目

---

## 一、项目定位

### 1.1 一句话

**"Tell it what to pick up — in plain English."**

用户用自然语言描述一个物体（"the red cup on the left side of the table"），系统在仿真环境中找到它、规划抓取位姿、执行抓取。

### 1.2 为什么能拿 Stars

| 因素 | 说明 |
|------|------|
| **Demo 惊艳** | 一个 Web 页面，输入文字，机器人就抓——所见即所得 |
| **技术栈现代** | Grounding DINO + SAM + MuJoCo——2024 组合 |
| **完整闭环** | 不是"又一个模型"，而是感知→规划→执行的完整系统 |
| **可复现** | 纯仿真，任何人在自己电脑上 10 分钟跑起来 |
| **有壁垒** | 开放词汇（open-vocabulary）抓取——不是固定几个物体的演示 |
| **Gradio 一键体验** | Hugging Face Spaces 上直接试——不用装任何东西 |
| **学术价值** | 可以作为 VLP/VLA 研究的 baseline |

### 1.3 对标项目（差距 = 机会）

| 项目 | Stars | 做对了什么 | 缺什么 |
|------|-------|-----------|--------|
| [CLIPort](https://github.com/cliport/cliport) | 500+ | 第一个把 CLIP 用于机器人 | 只支持 2D 桌面方块 |
| [VoxPoser](https://voxposer.github.io/) | 800+ | LLM 生成 3D 价值地图 | 需要真机，普通人跑不了 |
| [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) | 15k+ | Grounding DINO + SAM 组合 | 只有感知，没有动作 |
| [lerobot](https://github.com/huggingface/lerobot) | 8k+ | HuggingFace 背书 + 多机器人 | 聚焦模仿学习，不接地气 |

> **你的机会**：把 Grounded-SAM 的感知能力 + robosuite 的执行能力连起来，填补"从语言到抓取"的中间空白。

---

## 二、项目名称

**候选**：

| 名称 | 含义 | 投票 |
|------|------|------|
| **GraspGPT** | GPT-like grasping | ⭐⭐⭐ |
| **Lang2Grasp** | Language to Grasp | ⭐⭐⭐⭐ |
| **SayGrasp** | Say and Grasp | ⭐⭐⭐ |
| **VocabGrasp** | Open-Vocabulary Grasping | ⭐⭐ |
| **GraspAnything** | 抓取任何语言描述的东西 | ⭐⭐⭐⭐⭐ |

> 推荐 **GraspAnything** ——好记、好搜、直接传达"什么都能抓"。

---

## 三、系统架构

```
┌─────────────────────────────────────────────────────┐
│                   Gradio Web UI                      │
│   ┌─────────────┐   ┌──────────┐   ┌────────────┐  │
│   │ 文本输入     │   │ 摄像头   │   │ 抓取按钮   │  │
│   │ "red cup"   │   │ 实时画面 │   │ "Grasp It" │  │
│   └──────┬──────┘   └────┬─────┘   └─────┬──────┘  │
├──────────┼───────────────┼───────────────┼──────────┤
│          ▼               ▼               ▼          │
│  ┌─────────────────────────────────────────────┐   │
│  │              感知层 (Perception)              │   │
│  │  Grounding DINO → SAM → 3D 点云投影          │   │
│  │  文本→检测框      框→Mask    Mask→6D 位姿    │   │
│  └──────────────────────┬──────────────────────┘   │
│                         ▼                          │
│  ┌─────────────────────────────────────────────┐   │
│  │              规划层 (Planning)                │   │
│  │  抓取位姿生成 → 碰撞检测 → 运动规划            │   │
│  └──────────────────────┬──────────────────────┘   │
│                         ▼                          │
│  ┌─────────────────────────────────────────────┐   │
│  │              执行层 (Execution)               │   │
│  │  robosuite/MuJoCo → 机械臂轨迹 → 抓取         │   │
│  └─────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│              仿真环境 (MuJoCo + robosuite)           │
│  ┌──────────────────────────────────────────────┐  │
│  │  桌面场景：Franka Panda 手臂 + 10+ 随机物体    │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 四、技术选型

| 层 | 技术 | 理由 |
|----|------|------|
| **仿真** | MuJoCo + robosuite | pip 即装，物理引擎顶级，Ubuntu 24.04 原生支持 |
| **开放词汇检测** | Grounding DINO | 文本→检测框的最佳开源方案 |
| **分割** | SAM / EfficientSAM | 框→Mask，SAM 精度高，EfficientSAM 速度快 |
| **3D 投影** | 相机内参 + 深度图 | robosuite 提供深度图，投影到世界坐标 |
| **抓取规划** | 主方向估计 + 碰撞检测 | 基于 Mask 中心点和 PCA 主方向 |
| **Web UI** | Gradio | Hugging Face Spaces 直接部署 |
| **包管理** | uv / pip | 快 |
| **文档** | MkDocs Material | 专业感 |

---

## 五、项目结构

```
grasp-anything/
├── README.md                 ← 项目首页：动图 + 一键体验链接
├── docs/                     ← MkDocs 文档站
│   ├── index.md
│   ├── quickstart.md
│   ├── architecture.md
│   └── api.md
├── grasp_anything/           ← 核心 Python 包
│   ├── __init__.py
│   ├── perception.py         ← Grounding DINO + SAM 封装
│   ├── grounding.py          ← 文本→检测框
│   ├── segmentation.py       ← 框→Mask
│   ├── projection.py         ← Mask→3D 位姿
│   ├── grasping.py           ← 抓取规划
│   ├── robot.py              ← robosuite 交互封装
│   └── config.py             ← 配置管理
├── scripts/
│   ├── download_models.sh    ← 模型下载
│   └── check_install.py      ← 环境检查
├── app.py                    ← Gradio Web UI
├── demo.ipynb                ← Colab 一键运行
├── pyproject.toml            ← 项目配置
├── requirements.txt
└── .github/
    ├── workflows/            ← CI/CD
    └── ISSUE_TEMPLATE/
```

---

## 六、开发路线图

### Phase 1：最小可用（2 周）

> 目标：自己的电脑上跑通一条完整链路

- [ ] MuJoCo + robosuite 环境搭建，Franka Panda 手臂 + 桌面场景
- [ ] Grounding DINO 跑通——输入文本，输出检测框
- [ ] SAM 跑通——输入框，输出 Mask
- [ ] 简单抓取：Mask→中心点→固定方向下降抓取
- [ ] 验证：在仿真中成功抓取"red cube"

### Phase 2：打磨体验（1 周）

> 目标：Gradio Web 界面 + 一键安装

- [ ] Gradio 界面：文本框 + 仿真画面 + 抓取按钮
- [ ] `pip install -e .` 一键安装
- [ ] `python app.py` 一键启动
- [ ] 模型自动下载脚本

### Phase 3：提升质量（1 周）

> 目标：抓取成功率 > 80%，支持更多物体

- [ ] 桌面随机放置 10+ 种物体（YCB 物体集）
- [ ] 主方向估计（PCA 检测 Mask 旋转角度）
- [ ] 碰撞检测：避开已有物体
- [ ] 多视角：顶部→前视切换

### Phase 4：发布（1 周）

> 目标：GitHub 上线 + Hugging Face Spaces 部署

- [ ] MkDocs 文档站完整
- [ ] README 动图（抓取成功/失败的对比）
- [ ] Colab notebook 一键运行
- [ ] Hugging Face Spaces 部署
- [ ] 发布 Twitter/X + Reddit + 知乎 + 即刻
- [ ] 提交到 GitHub Trending + Papers With Code

---

## 七、传播策略

### 7.1 发布的时机

| 时间 | Star 潜力 | 原因 |
|------|----------|------|
| 有 Gradio demo | ⭐⭐ | 只有截图，大家 star 了也不会试 |
| Hugging Face Spaces 上线 | ⭐⭐⭐⭐ | 一键试玩，无需安装 |
| Colab 可复现 | ⭐⭐⭐ | 开发者会真的 fork 和引用 |
| 有对比实验 | ⭐⭐⭐⭐⭐ | 和 CLIPort / VoxPoser 做定量对比 |

### 7.2 README 要有的东西

1. **首屏动图**——用户输入"red cup"，机器人抓起来的全过程 GIF
2. **Hugging Face Spaces 链接**——"Try it now, no install required"
3. **pip install**——3 行命令搞定
4. **定性结果**——6 组抓取截图（成功 vs 失败）
5. **定量结果**——对比表格

### 7.3 发布渠道

| 渠道 | 受众 | 发布内容 |
|------|------|---------|
| Twitter/X | 全球 AI/ML 研究者 | 抓取动图 + Spaces 链接 |
| Reddit r/MachineLearning | 开发者 | 技术细节 + 开源代码 |
| 知乎 | 中国开发者 | "我做了一个能用自然语言抓取任何物体的开源项目" |
| 即刻/V2EX | 中国工程师 | 轻量分享 |

---

## 八、成功的定义

| 阶段 | 指标 | 时间 |
|------|------|------|
| Launch | 100 stars | 第 1 周 |
| Traction | 500 stars | 第 1 个月 |
| Success | 1000+ stars | 第 3 个月 |
| Moonshot | 3000+ stars | 第 6 个月 |

GraspAnything 不需要成为 SOTA 论文，它只要**比 CLIPort 更好上手、比 VoxPoser 更可复现、比 Grounded-SAM 多一步执行**——就赢了。

---

*项目目录：`/home/amanannn/Projects/grasp-anything/`（建议另开仓库，不放在 Embodied-AI-Lab 里）*
