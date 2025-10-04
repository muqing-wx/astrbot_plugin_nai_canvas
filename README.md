# NovelAI 智能绘图插件 (v1.0.0)

本插件基于 NovelAI 官方及第三方 API，为您的 AstrBot 提供了强大且高度可定制的 AI 绘图功能。插件内置了先进的 **LLM 智能提示词处理系统**，能够自动分析用户意图，将简单的自然语言描述（如“一个女孩”）智能地扩写或翻译为专业、丰富的英文绘图标签，极大地降低了使用门槛并提升了出图质量。

## ✨ 功能特性

-   **智能提示词处理 (LLM)**:
    -   **自动分析**: 插件能智能区分用户的输入是“简单描述”、“详细描述”还是“专业提示词”，并采取最优处理策略。
    -   **创意扩写**: 对简单的想法（如“夜晚的城市”）进行创意性的细节补充和标签扩展。
    -   **精准翻译**: 将详细的中文场景描述准确翻译为符合 AI 绘图习惯的英文标签。
    -   **完全可配**: 用户可以自由配置用于提示词处理的 LLM 模型、API 地址及密钥。

-   **双 API 通道支持**:
    -   **官方通道**: 支持标准的 NovelAI 官方 API (POST 请求)。
    -   **第三方代理**: 兼容社区常见的第三方代理服务 (GET 请求)，如 `std.loliyc.com`。

-   **高度可定制化**:
    -   提供丰富的绘图参数配置，包括模型、采样器、分辨率、步数、CFG 等。
    -   支持自定义图像尺寸，满足各种构图需求。

-   **画师/风格预设管理**:
    -   用户可以创建、查看和管理多套正/反向提示词预设（画师串）。
    -   通过简单的指令即可在绘图时调用指定的预设。

-   **智能密钥轮询**:
    -   支持配置多个 NAI API 密钥和 LLM API 密钥。
    -   当某个密钥失效或请求失败时，插件会自动切换到下一个可用的密钥，保证服务稳定性。

-   **灵活的内容控制**:
    -   可配置是否默认生成 NSFW 内容，插件会根据设置智能地添加或移除 `nsfw` 标签。

## 📦 安装说明

1.  **下载插件文件**:
    确保您拥有 `main.py`, `_conf_schema.json`, `metadata.yaml`, `requirements.txt` 这四个文件。

2.  **放置文件**:
    在您的 AstrBot 根目录下，找到 `data/plugins` 文件夹。在其中创建一个新的文件夹，例如 `astrbot_plugin_nai_canvas`。
    将上述四个文件放入刚刚创建的文件夹中。

3.  **安装依赖**:
    在您的 AstrBot 环境中，执行以下命令来安装插件所需的依赖库：
    ```bash
    pip install -r data/plugins/astrbot_plugin_nai_canvas/requirements.txt
    ```
    或者手动安装：
    ```bash
    pip install aiohttp
    ```

4.  **重启 AstrBot**:
    完全关闭并重新启动您的 AstrBot 程序，插件将自动加载。

## ⚙️ 配置说明

插件加载后，会在 `data/configs` 目录下生成一个名为 `astrbot_plugin_nai_canvas.json` 的配置文件。请根据您的需求打开并编辑它。

### API 相关配置

| 配置项 (Key) | 类型 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- |
| `api_channel` | string | API 请求通道。选择“官方”使用 POST 请求；选择“第三方代理”使用 GET 请求。 | `"官方 (official)"` |
| `nai_api_keys` | list | **【必需】** 您的 NovelAI API 密钥 (Token) 列表。支持多个，失效时自动切换。 | `[]` |
| `third_party_api_endpoint` | string | 当通道为“第三方代理”时使用的 API 地址。 | `"https://std.loliyc.com/generate"` |
| `third_party_disable_cache` | bool | 禁用第三方 API 缓存。开启后可确保随机性，但可能稍慢。 | `true` |

### LLM (提示词增强) 配置

| 配置项 (Key) | 类型 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- |
| `enable_prompt_enhancement` | bool | 是否启用 LLM 对用户提示词进行智能处理。 | `true` |
| `llm_api_keys` | list | **【必需】** 用于提示词增强的 LLM API 密钥列表 (兼容 OpenAI/SiliconFlow)。 | `[]` |
| `llm_api_base_url` | string | LLM API 的基础 URL 地址。 | `"https://api.siliconflow.cn/v1"` |
| `llm_model_name` | string | 用于处理提示词的 LLM 模型名称。 | `"Qwen/Qwen2-7B-Instruct"` |

### 绘图参数配置

| 配置项 (Key) | 类型 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- |
| `model` | string | 选择 NovelAI 绘图模型。 | `"nai-diffusion-4-5-full"` |
| `sampler` | string | 选择采样算法。 | `"k_dpmpp_2m"` |
| `noise_schedule` | string | 选择噪声调度算法。 | `"karras"` |
| `resolution_preset` | string | 预设的图片尺寸。 | `"竖图 (832x1216)"` |
| `custom_width` | int | 当预设尺寸为“自定义”时生效的宽度。 | `1920` |
| `custom_height` | int | 当预设尺寸为“自定义”时生效的高度。 | `1088` |
| `steps` | int | 步数，推荐 28-50。 | `28` |
| `scale` | float | 提示词引导值 (CFG)，推荐 5-7。 | `5.0` |
| `unclip_guidance_scale` | float | 缩放引导值，V3 模型新增参数，建议为 0。 | `0.0` |
| `seed` | int | 种子，用于复现图像。0 表示随机。 | `0` |
| `smea` | bool | 是否启用 SMEA 采样器优化。 | `false` |
| `smea_dyn` | bool | 是否启用 SMEA DYN 动态版本。 | `false` |

### 其他配置

| 配置项 (Key) | 类型 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- |
| `save_images_locally` | bool | 是否在本地 `temp_images` 文件夹中永久保存生成的图片。 | `false` |
| `enable_nsfw_by_default` | bool | 是否默认生成 NSFW 内容。插件会根据此选项智能调整 `nsfw` 标签。 | `false` |

## 🚀 使用指令

### 核心绘图指令

-   `/nai生图 [预设名] <内容>`
    -   **说明**: 如果不指定`[预设名]`，将自动使用“默认”预设。插件会根据`<内容>`的格式，智能选择处理模式。
    -   **智能模式 (自然语言)**: 当输入是“一个穿白裙的女孩在海边”时，插件会进行翻译和标签化，并与预设的**正向**提示词**融合**。
    -   **专业模式 (专业标签)**: 当输入是`1girl, masterpiece | lowres`时，插件会用其**覆盖**预设的提示词。
        -   `正向`: 覆盖预设正向，使用预设反向。
        -   `正向 | 反向`: 同时覆盖预设的正向和反向。
        -   `| 反向`: 使用预设正向，覆盖预设反向。

### 提示词预设管理指令

-   `/nai增加提示词 <名称>|<正向>|<反向>` (仅限机器人所有者)
-   `/nai删除提示词 <名称>` (仅限机器人所有者)
-   `/nai提示词列表`
-   `/nai查看提示词 <名称>`
-   `/nai帮助`

## ❓ 常见问题 (FAQ)

1.  **机器人提示 "未配置 NAI API 密钥"**
    > 请检查配置文件 `astrbot_plugin_nai_canvas.json`，确保 `nai_api_keys` 字段已正确填写您的 NovelAI 密钥。

2.  **智能模式/提示词增强不生效**
    > 请检查配置文件中 `enable_prompt_enhancement` 是否为 `true`，并确保 `llm_api_keys`, `llm_api_base_url`, `llm_model_name` 已正确配置且密钥有效。

3.  **第三方代理无法使用**
    > 请确保配置文件中的 `api_channel` 设置为 `"第三方代理 (third_party)"`，并且 `third_party_api_endpoint` 地址是正确的。

4.  **如何使用画师串？**
    > 首先，使用 `/nai增加提示词` 命令将您的画师串保存为一个预设。例如：`/nai增加提示词 Rella串|Artist:rella...|lowres...`。然后，在绘图时调用它：`/nai生图 Rella串 一个女孩`。
