# Qwen3.5-35B-A3B 模型介绍

## 概述

Qwen3.5-35B-A3B 是阿里巴巴千问团队于2026年2月25日开源的中等规模大语言模型。作为 Qwen3.5 系列的重要成员，该模型采用创新的混合架构，在中等参数规模下实现了卓越的性能表现，成功登顶 Hugging Face 趋势榜。

> **📝 说明**: 模型架构图可以从 Hugging Face 官方页面查看：https://huggingface.co/Qwen/Qwen3.5-35B-A3B

## 核心特性

### 1. 创新架构设计

Qwen3.5-35B-A3B 采用了 **Mixture of Experts (MoE)** 架构，具有以下特点：

- **总参数量**: 35B (350 亿参数)
- **激活参数**: 每次前向传播仅激活约 3B 参数
- **专家数量**: 256 个专家中选择 9 个进行计算
- **稀疏激活**: 大幅降低推理成本和延迟

这种设计使得模型在保持强大能力的同时，显著提升了推理效率和性价比。

### 2. 性能表现

Qwen3.5-35B-A3B 在性能方面表现卓越：

- **超越前代**: 性能已超过 Qwen3.2-35B-A22B-2507 和 Qwen3-VL-35B-A22B 等更大规模的模型
- **对比 GPT**: 在多项基准测试中表现优于 GPT-5 mini
- **Hugging Face**: 发布不到 24 小时即登顶 Hugging Face 趋势榜
- **中等规模标杆**: 创下中等尺寸模型性能新高

### 3. 硬件适配性

模型在不同硬件平台上的表现：

#### Mac 平台实测

在 **Apple Mac Studio (M1 Ultra)** 上的测试结果：

- **配置**: M1 Ultra, 64GB RAM, 20核CPU, 48核GPU
- **模型版本**: Qwen3.5-35B-A3B-4bit
- **生成速度**: **60 tokens/秒**
- **评价**: 刷新了对 4bit 量化模型的认知

#### AMD GPU 测试

- **设备**: AMD Radeon RX 5700 XT (gfx906)
- **量化版本**: Q4_K_Medium
- **模型大小**: 18.32 GiB
- **速度**: 841.72 ± 4.31 tokens/s (pp512, ngl=99)

## 技术规格

### 参数规模

| 参数类型 | 大小 | 说明 |
|---------|------|------|
| 总参数 | 35B | 模型的全部参数量 |
| 激活参数 | ~3B/token | 每次推理实际激活的参数 |
| 专家数量 | 256 | MoE 架构中的专家总数 |
| 激活专家数 | 9 | 每次推理使用的专家数量 |

### 模型文件大小

| 精度 | 大小 | 说明 |
|------|------|------|
| FP16 | ~70GB | 半精度浮点数 |
| FP8 | ~35GB | 8位浮点数 |
| NVFP4 | ~18GB | NVIDIA 4位量化 |
| GGUF Q8 | ~37GB | 8位量化版本 |

### 架构特点

Qwen3.5-35B-A3B 的混合架构结合了：

1. **Gated Delta Networks (GDN)**: 线性注意力机制
2. **Sparse Mixture of Experts (MoE)**: 稀疏混合专家模型

这种混合设计在以下方面实现平衡：

- ✅ **能力保持**: 与密集模型相当的推理能力
- ✅ **速度优化**: 稀疏激活降低计算量
- ✅ **成本控制**: 更高的推理效率

## 部署方案

### 1. vLLM 部署

vLLM 是高效的推理引擎，适合生产环境部署：

```bash
# 安装 vLLM
pip install vllm

# 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-35B-A3B \
    --trust-remote-code \
    --tensor-parallel-size 2
```

### 2. SGLang 部署

SGLang 优化了结构化生成和长文本处理：

```bash
# 安装 SGLang
pip install "sglang[all]"

# 启动服务
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B \
    --tp 2 \
    --port 8000
```

### 3. Ollama 部署

使用 Ollama 轻量部署（适合本地开发）：

```bash
# 下载模型
ollama pull qwen3.5:35b

# 运行模型
ollama run qwen3.5:35b

# 测试
ollama run qwen3.5:35b "你好，请介绍一下你自己"
```

### 4. llama.cpp 部署

使用 GGUF 格式在 CPU 上运行：

```bash
# 下载 GGUF 版本
wget https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf

# 使用 llama.cpp 运行
./llama-cli -m Qwen3.5-35B-A3B-Q4_K_M.gguf -p "你好" -n 512
```

## 硬件需求

### 最低配置

| 组件 | 要求 | 说明 |
|------|------|------|
| GPU 显存 | 18GB | 使用 NVFP4 量化 |
| 系统内存 | 32GB | CPU 推理 |
| 存储空间 | 40GB | 模型文件 + 临时文件 |

### 推荐配置

| 组件 | 要求 | 说明 |
|------|------|------|
| GPU 显存 | 48GB | 使用 FP8 精度 |
| 系统内存 | 64GB | 更好的吞吐量 |
| GPU | NVIDIA 4090/5090 或同类 | 推荐使用 CUDA |

### 集群部署 (NVIDIA DGX Spark)

DGX Spark 集群规格：

| 组件 | 规格 |
|------|------|
| 处理器 | GB10 Grace Blackwell 超级芯片 (ARM aarch64) |
| 内存 | 128GB LPDDR5x 统一内存 |
| 内存带宽 | 273 GB/s |
| AI 算力 | 1 petaFLOP (FP4 精度) |
| 网络 | 双端口 ConnectX-7 (200 Gbps RoCE) |
| 存储 | 4TB NVMe |

## 使用示例

### Python 代码示例

#### 使用 Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3.5-35B-A3B"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

# 生成文本
prompt = "请介绍一下人工智能的发展历程"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

#### 使用 vLLM API

```python
from openai import OpenAI

# 连接到 vLLM 服务
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 调用模型
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-35B-A3B",
    messages=[
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

### 工具调用示例

Qwen3.5-35B-A3B 支持函数调用：

```python
from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 调用
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-35B-A3B",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools
)

print(json.dumps(response.choices[0].message.tool_calls, indent=2))
```

## 性能优化建议

### 1. 量化策略

根据硬件条件选择合适的量化方式：

| 硬件配置 | 推荐精度 | 模型大小 | 预期速度 |
|---------|---------|---------|---------|
| 16GB 显存 | Q4_K_M | ~18GB | 中等 |
| 24GB 显存 | Q6_K | ~26GB | 较快 |
| 48GB+ 显存 | FP8 | ~35GB | 最快 |

### 2. 推理优化

- **批处理**: 合并多个请求以提高吞吐量
- **KV Cache**: 启用以减少重复计算
- **流式输出**: 降低首字延迟
- **Tensor Parallel**: 多 GPU 并行加速

### 3. 部署优化

```python
# vLLM 优化配置
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-35B-A3B \
    --quantization awq \  # 使用 AWQ 量化
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --dtype auto
```

## 应用场景

Qwen3.5-35B-A3B 适用于以下场景：

### 1. 智能代理 (Agentic AI)

- 复杂任务规划和执行
- 多步推理和工具调用
- 自主决策和反思

### 2. 代码生成

- 代码编写和补全
- 代码审查和优化建议
- 技术文档生成

### 3. 知识问答

- 专业知识查询
- 教育和学习辅助
- 研究文献总结

### 4. 内容创作

- 文章写作
- 创意生成
- 多语言翻译

### 5. 数据分析

- 数据解读
- 趋势分析
- 报告生成

## 模型下载

### Hugging Face

```bash
# 完整模型
git lfs install
git clone https://huggingface.co/Qwen/Qwen3.5-35B-A3B

# 或使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3.5-35B-A3B
```

### Ollama

```bash
ollama pull qwen3.5:35b
```

### GGUF 量化版本

```bash
# 从 unsloth 仓库下载
wget https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf
```

## 常见问题 (FAQ)

### Q1: Qwen3.5-35B-A3B 需要多大的显存？

**A**:
- FP16 精度: ~70GB
- FP8 精度: ~35GB
- NVFP4 量化: ~18GB
- Q4 量化: ~18GB

### Q2: 可以在 Mac 上运行吗？

**A**: 可以！实测在 M1 Ultra (64GB RAM) 上使用 4bit 量化版本可达到 60 tokens/秒。

### Q3: 模型支持哪些语言？

**A**: Qwen3.5 系列主要针对中文和英文优化，但也支持多语言。

### Q4: 如何选择量化版本？

**A**:
- **Q4_K_M**: 平衡精度和速度，推荐首选
- **Q6_K**: 更高精度，显存充足时使用
- **Q8_K**: 接近原版精度，需要更多资源

### Q5: 模型支持工具调用吗？

**A**: 支持，可以像使用 OpenAI API 一样进行函数调用。

## 版本对比

### Qwen 系列演进

| 模型 | 参数量 | 架构 | 特点 |
|------|--------|------|------|
| Qwen2.5-32B | 32B | Dense | 基础版本 |
| Qwen3.2-35B-A22B | 35B (激活22B) | MoE | 初代 MoE |
| Qwen3.5-35B-A3B | 35B (激活3B) | MoE | 性能大幅提升 |
| Qwen3.5-122B-A10B | 122B (激活10B) | MoE | 更大模型 |

### 性能提升

Qwen3.5-35B-A3B 相比前代：

- ✅ **推理速度**: 提升 3-5x (稀疏激活)
- ✅ **显存占用**: 降低 60% (激活3B vs 35B)
- ✅ **能力保持**: 不降反升，超越更大模型
- ✅ **成本效率**: 更高的性价比

## 社区资源

### 官方资源

- 🤗 **Hugging Face**: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- 📖 **论文**: [待补充论文链接]
- 💬 **Discord**: Qwen 官方社区
- 🐙 **GitHub**: https://github.com/QwenLM/Qwen

### 第三方项目

- 🚀 **vLLM**: https://github.com/vllm-project/vllm
- ⚡ **SGLang**: https://github.com/sgl-project/sglang
- 🦙 **llama.cpp**: https://github.com/ggerganov/llama.cpp
- 🫙 **Ollama**: https://ollama.ai

## 许可证

Qwen3.5-35B-A3B 采用 **Qwen License**，具体使用条款请参考 Hugging Face 模型页面。

## 致谢

感谢 Qwen 团队和所有贡献者的开源贡献！

---

**最后更新**: 2026年2月
**文档版本**: 1.0
