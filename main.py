import base64
import os
import uuid
import io
import zipfile
import json
import re
import random
from pathlib import Path
import textwrap
from urllib.parse import urlencode

import aiohttp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.message_components import Image
from astrbot.api.star import Star, register


@register("astrbot_plugin_nai_canvas", "沐沐沐倾", "NovelAI 智能绘图", "1.0.0")
class NAICanvas(Star):
    """
    基于 NovelAI API 的智能绘图插件，提供丰富的参数自定义选项，让你轻松创作高质量AI绘画作品。
    """
    # --- 内置LLM提示词模板 ---
    LLM_UNIFIED_JUDGEMENT_TEMPLATE = textwrap.dedent("""
        **角色定义：** 你是一个AI绘画提示词分析器，以JSON API的形式工作。你的**唯一**任务是分析用户输入的**正向提示词**，并将其结构化为指定的JSON格式。

        **任务描述：**
        根据用户输入的**意图和内容结构**，将其精确分类到以下三种处理策略之一，并提取相应内容。

        **处理策略定义 (按此顺序判断)：**
        1.  **简单描述 (Strategy: `expand`)**: 输入是只包含一个**核心主体**和少量修-饰词的自然语言短语，缺乏场景、构图等细节。
            *   **处理方式：** 提取整个短语用于后续的创意扩写。
            *   **示例输入:** '一个女孩', '夜晚的城市', 'a cute catgirl'
            *   **输出JSON:** `{\"processing_strategy\": \"expand\", \"content\": {\"prompt\": \"a cute catgirl\"}}`

        2.  **详细描述 (Strategy: `translate_and_tagify`)**: 输入是描述了**具体场景、人物、动作、服装**等丰富细节的自然语言句子。
            *   **处理方式：** 提取整个句子用于后续的翻译和标签化。
            *   **示例输入:** '一个穿着白色连衣裙的女孩在雨中散步'
            *   **输出JSON:** `{\"processing_strategy\": \"translate_and_tagify\", \"content\": {\"prompt\": \"一个穿着白色连衣裙的女孩在雨中散步\"}}`

        3.  **专业提示词 (Strategy: `process_directly`)**: 输入包含大量逗号分隔的**英文标签**或特殊权重语法（如`::`, `{}`, `[]`）。
            *   **处理方式：** 直接提取整个输入内容。
            *   **示例输入:** `masterpiece, best quality, 1girl, a girl in the rain`
            *   **输出JSON:** `{\"processing_strategy\": \"process_directly\", \"content\": {\"prompt\": \"masterpiece, best quality, 1girl, a girl in the rain\"}}`

        **输出格式强制要求：**
        - 你的回复**必须**是一个**纯净的、不含任何杂质的JSON对象**。
        - **绝对禁止**在JSON对象前后添加任何解释、注释或代码块标记。

        **立即处理以下用户输入:** '{{original_prompt}}'
    """)

    LLM_TRANSLATION_TEMPLATE = textwrap.dedent("""
        **角色定义：** 你是一个专用的、无状态的翻译引擎。你的**唯一功能**是将输入的中文自然语言，转换为用于AI绘画的、**逗号分隔的英文标签**。
        **严格规则：**
        1. **输出纯净：** 你的输出**必须**只包含英文标签，并用 `, ` (逗号加空格) 分隔。
        2. **禁止对话：** **绝对禁止**输出任何形式的句子、解释、前缀或任何非标签内容。
        3. **忠于原文：** **只翻译**用户描述的核心概念，**禁止**自行添加任何画质、风格或无关的标签。
        4. **绝对禁止格式化：** 你的输出**绝对不能**包含任何Markdown标记（如反引号 ` `）、代码块或任何非标签字符。
        **立即将以下文本转换为英文标签:** '{{original_prompt}}'
    """)

    LLM_EXPANSION_TEMPLATE = textwrap.dedent("""
        **角色定义：** 你是一个高度自律的AI绘画提示词生成器。你的**唯一任务**是将一个简单的英文核心概念，扩展成一组丰富、详细、且**纯粹由逗号分隔的英文标签**组成的字符串，风格偏向动漫/插画。
        **输出强制要求：**
        1. **格式纯粹：** 你的回复**必须**直接以第一个英文标签开始，以最后一个英文标签结束。**只能**包含英文标签和用于分隔的 `, `。
        2. **绝对禁止格式化：** **严禁**包含任何非标签内容，如前缀、后缀、解释、句子、标题、Markdown反引号、代码块标记。
        **立即将以下核心概念扩展为一组丰富的英文标签:** '{{original_prompt}}'
    """)

    # --- 插件常量 ---
    OFFICIAL_API_ENDPOINT_URL = "https://api.novelai.net/ai/generate-image"
    API_CHANNEL_MAP = {"官方 (official)": "official", "第三方代理 (third_party)": "third_party"}
    RESOLUTION_MAP = {
        "竖图 (832x1216)": {"height": 1216, "width": 832, "size_str": "竖图"},
        "横图 (1216x832)": {"height": 832, "width": 1216, "size_str": "横图"},
        "方图 (1024x1024)": {"height": 1024, "width": 1024, "size_str": "方图"},
        "自定义 (custom)": {"height": None, "width": None, "size_str": "自定义"}
    }

    def __init__(self, context, config):
        super().__init__(context)
        self.config = config
        logger.info("NovelAI 智能绘图插件初始化...")

        self.plugin_dir = Path(__file__).parent
        self.save_dir = self.plugin_dir / "temp_images"
        self.presets_file = self.plugin_dir / "user_presets.json"
        self.save_dir.mkdir(exist_ok=True)

        # --- 从配置加载 ---
        api_channel_display = self.config.get("api_channel", "官方 (official)")
        self.api_channel = self.API_CHANNEL_MAP.get(api_channel_display, "official")
        self.nai_api_keys = self.config.get("nai_api_keys", [])
        self.nai_current_key_index = 0
        self.third_party_api_endpoint_config = self.config.get("third_party_api_endpoint", "https://std.loliyc.com/generate")
        self.third_party_disable_cache = self.config.get("third_party_disable_cache", True)
        self.save_images_locally = self.config.get("save_images_locally", False)
        self.enable_nsfw_by_default = self.config.get("enable_nsfw_by_default", False)

        self.model = self.config.get("model", "nai-diffusion-4-5-full")
        self.sampler = self.config.get("sampler", "k_dpmpp_2m")
        self.noise_schedule = self.config.get("noise_schedule", "karras")

        resolution_display = self.config.get("resolution_preset", "竖图 (832x1216)")
        self.resolution_preset_data = self.RESOLUTION_MAP.get(resolution_display, self.RESOLUTION_MAP["竖图 (832x1216)"])
        self.custom_width = self.config.get("custom_width", 1024)
        self.custom_height = self.config.get("custom_height", 1024)

        self.steps = self.config.get("steps", 28)
        self.scale = self.config.get("scale", 5.0)
        self.unclip_guidance_scale = self.config.get("unclip_guidance_scale", 0.0)
        self.seed = self.config.get("seed", 0)
        self.smea = self.config.get("smea", False)
        self.smea_dyn = self.config.get("smea_dyn", False)

        self.enable_prompt_enhancement = self.config.get("enable_prompt_enhancement", True)
        self.llm_api_keys = self.config.get("llm_api_keys", [])
        self.llm_current_key_index = 0
        self.llm_api_base_url = self._normalize_api_base_url(self.config.get("llm_api_base_url", "https://api.siliconflow.cn/v1"))
        self.llm_model_name = self.config.get("llm_model_name", "Qwen/Qwen2-7B-Instruct")

        self.presets = self._load_presets()

    def is_admin(self, event: AstrMessageEvent) -> bool:
        return event.get_sender_id() in self.context.get_config().get("admins_id", [])

    def _load_presets(self):
        default_preset = {
            "默认": {
                "positive": "2::official art, year2024, year2025 ::,1.85::Artist:youngjoo kjy ::,1.35::Artist:zer0.zer0 ::,1.15::Artist:stu_dts ::,1.15::artist:ogipote ::,1.05::Artist:qiandaiyiyu ::,1.25::Artist:rella ::,1.05::Artist:atdan ::,0.85::artist:hiten (hitenkei)::,0.65::Artist:ask_(askzy) ::,0.75::Artist:nixeu ::,-3::3D ::,-1.5::artist collaboration ::,1.35::rim lighting, deep shadows,volumetric lighting,high contrast, cinematic lighting ::, {no text,realistic, 8k }, 1.63::photorealistic::, 20::best quality, absurdres, very aesthetic, detailed, masterpiece::, assisted exposure, looking at viewer, no text",
                "negative": "nsfw, lowres, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, dithering, halftone, screentone, multiple views, logo, too many watermarks, negative space, blank page, worst quality,low quality,artist collaboration, bad anatomy,extra fingers,extra legs, missing legs, missing fingers, mutation, text, watermark, low resolution"
            }
        }
        if not self.presets_file.exists():
            self._save_presets(default_preset)
            return default_preset
        try:
            with open(self.presets_file, 'r', encoding='utf-8') as f:
                presets = json.load(f)
                if "默认" not in presets: presets.update(default_preset)
                return presets
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"加载提示词文件失败: {e}")
            return default_preset

    def _save_presets(self, presets_data):
        try:
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                json.dump(presets_data, f, ensure_ascii=False, indent=4)
            return True
        except IOError as e:
            logger.error(f"保存提示词文件失败: {e}")
            return False

    def _normalize_api_base_url(self, url):
        return url.strip().rstrip('/')

    def _get_current_nai_api_key(self):
        if not self.nai_api_keys: return None
        return self.nai_api_keys[self.nai_current_key_index]

    def _switch_next_nai_api_key(self):
        if not self.nai_api_keys: return
        self.nai_current_key_index = (self.nai_current_key_index + 1) % len(self.nai_api_keys)
        logger.info(f"NAI密钥失效或请求失败，切换到下一个Key (索引: {self.nai_current_key_index})")

    async def _call_nai_api(self, payload):
        if not self.nai_api_keys: raise Exception("未配置 NAI API 密钥。")
        max_attempts = len(self.nai_api_keys)
        for attempt in range(max_attempts):
            current_key = self._get_current_nai_api_key()
            if not current_key: self._switch_next_nai_api_key(); continue
            
            async with aiohttp.ClientSession() as session:
                try:
                    if self.api_channel == "third_party":
                        endpoint = self.third_party_api_endpoint_config
                        width, height = self.get_dimensions()
                        
                        if self.resolution_preset_data["size_str"] == "自定义":
                            size_str = f"{width}x{height}"
                        else:
                            size_str = self.resolution_preset_data["size_str"]
                        
                        params = {
                            "token": current_key, "model": self.model, "sampler": self.sampler,
                            "steps": self.steps, "scale": self.scale, "seed": payload["parameters"]["seed"],
                            "noise_schedule": self.noise_schedule, "size": size_str,
                            "tag": payload.get("input", ""), "uc": payload.get("parameters", {}).get("uc", ""),
                            "cfg": self.unclip_guidance_scale, 
                            "nocache": 1 if self.third_party_disable_cache else 0
                        }
                        params = {k: v for k, v in params.items() if v is not None}
                        logger.info(f"请求第三方 GET API (尝试 {attempt + 1}/{max_attempts})...")
                        async with session.get(url=endpoint, params=params) as response:
                            if response.status == 200 and 'image/' in response.headers.get('Content-Type', ''):
                                logger.info("第三方 API 图片数据接收成功。")
                                return await response.read()
                            else:
                                error_text = await response.text()
                                logger.warning(f"第三方 API 请求失败 ({response.status}): {error_text}")
                                self._switch_next_nai_api_key()
                    else:
                        endpoint = self.OFFICIAL_API_ENDPOINT_URL
                        headers = {"Authorization": f"Bearer {current_key}", "Content-Type": "application/json"}
                        logger.info(f"请求官方 POST API (尝试 {attempt + 1}/{max_attempts})...")
                        async with session.post(url=endpoint, json=payload, headers=headers) as response:
                            if response.status == 200 and 'application/zip' in response.headers.get('Content-Type', ''):
                                logger.info("官方 API 压缩包数据接收成功。")
                                return await response.read()
                            error_text = await response.text(); error_message = error_text
                            try: error_message = json.loads(error_text).get("message", error_text)
                            except json.JSONDecodeError: pass
                            logger.warning(f"官方 API 请求失败 ({response.status}): {error_message}")
                            self._switch_next_nai_api_key()
                except aiohttp.ClientError as e:
                    logger.warning(f"NAI API 网络请求失败: {e}"); self._switch_next_nai_api_key()
        raise Exception("所有 NAI API 密钥均尝试失败。")

    def _get_current_llm_api_key(self):
        if not self.llm_api_keys: return None
        return self.llm_api_keys[self.llm_current_key_index]

    def _switch_next_llm_api_key(self):
        if not self.llm_api_keys: return
        self.llm_current_key_index = (self.llm_current_key_index + 1) % len(self.llm_api_keys)
        logger.info(f"LLM密钥失效或请求失败，切换到下一个Key (索引: {self.llm_current_key_index})")

    async def _call_llm_api(self, payload):
        if not self.llm_api_keys: raise Exception("未配置 LLM API 密钥。")
        endpoint = f"{self.llm_api_base_url}/chat/completions"
        max_attempts = len(self.llm_api_keys)
        for attempt in range(max_attempts):
            current_key = self._get_current_llm_api_key()
            if not current_key: self._switch_next_llm_api_key(); continue
            headers = {"Authorization": f"Bearer {current_key}", "Content-Type": "application/json"}
            if "model" not in payload: payload["model"] = self.llm_model_name
            logger.info(f"请求 LLM API (尝试 {attempt + 1}/{max_attempts})...")
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(url=endpoint, json=payload, headers=headers) as response:
                        response.raise_for_status()
                        data = await response.json()
                        if data.get("choices"): return data["choices"][0]["message"]["content"]
                        raise Exception("LLM API 响应格式不符。")
                except Exception as e:
                    logger.warning(f"LLM API 请求失败: {e}"); self._switch_next_llm_api_key()
        raise Exception("所有 LLM API 密钥均尝试失败。")

    def _clean_and_parse_json(self, response_text: str) -> dict:
        clean_json_str = response_text
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                clean_json_str = json_match.group(0)
        
        try:
            return json.loads(clean_json_str)
        except json.JSONDecodeError:
            logger.error(f"LLM返回的不是有效的JSON, 清理后内容: {clean_json_str}")
            raise ValueError("LLM返回格式错误，无法解析JSON。")

    def _clean_llm_output(self, text: str) -> str:
        """极度强化清洗LLM返回的文本，去除所有可能的污染。"""
        cleaned = text.replace('"', '').replace("'", "").replace("`", "")
        cleaned = cleaned.replace('，', ',')
        return cleaned.strip()

    async def _get_llm_analysis(self, prompt):
        llm_prompt = self.LLM_UNIFIED_JUDGEMENT_TEMPLATE.replace("{{original_prompt}}", prompt)
        payload = {"messages": [{"role": "user", "content": llm_prompt}], "temperature": 0.0, "response_format": {"type": "json_object"}}
        response_text = await self._call_llm_api(payload)
        return self._clean_and_parse_json(response_text)

    async def _translate_chinese_prompt(self, prompt):
        llm_prompt = self.LLM_TRANSLATION_TEMPLATE.replace("{{original_prompt}}", prompt)
        payload = {"messages": [{"role": "user", "content": llm_prompt}], "temperature": 0.2}
        response_text = await self._call_llm_api(payload)
        return self._clean_llm_output(response_text)

    async def _expand_simple_prompt(self, prompt):
        llm_prompt = self.LLM_EXPANSION_TEMPLATE.replace("{{original_prompt}}", prompt)
        payload = {"messages": [{"role": "user", "content": llm_prompt}], "temperature": 0.7}
        response_text = await self._call_llm_api(payload)
        return self._clean_llm_output(response_text)

    def get_dimensions(self):
        if self.resolution_preset_data["width"] is None or self.resolution_preset_data["height"] is None:
            return self.custom_width, self.custom_height
        return self.resolution_preset_data["width"], self.resolution_preset_data["height"]

    async def _process_mixed_prompt(self, prompt_str: str) -> str:
        """处理可能混合中英文的提示词，返回纯净的英文标签字符串。"""
        if not prompt_str:
            return ""
        
        normalized_prompt = self._clean_llm_output(prompt_str)
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]+', normalized_prompt)
        if not chinese_chars:
            return normalized_prompt

        chinese_to_translate = ' '.join(chinese_chars)
        translated_tags = await self._translate_chinese_prompt(chinese_to_translate)
        non_chinese_part = re.sub(r'[\u4e00-\u9fa5]+', '', normalized_prompt)
        
        parts = [p.strip() for p in [non_chinese_part, translated_tags] if p.strip()]
        return ', '.join(parts)

    async def _generate_image_task(self, event, positive_prompt, negative_prompt, apply_nsfw_logic=True, apply_comma_formatting=True):
        try:
            if apply_nsfw_logic:
                positive_prompt_lower = positive_prompt.lower()
                has_nsfw_in_positive = 'nsfw' in positive_prompt_lower

                if not self.enable_nsfw_by_default:
                    if has_nsfw_in_positive:
                        positive_prompt = re.sub(r'\bnsfw\b', '', positive_prompt, flags=re.IGNORECASE)
                        logger.warning("插件配置未允许NSFW，已从正向提示词中移除'nsfw'标签。")
                    if 'nsfw' not in negative_prompt.lower():
                        negative_prompt = f"nsfw, {negative_prompt}" if negative_prompt else "nsfw"
                else:
                    if not has_nsfw_in_positive:
                        positive_prompt = f"nsfw, {positive_prompt}" if positive_prompt else "nsfw"

            if apply_comma_formatting:
                positive_prompt = ', '.join([p.strip() for p in positive_prompt.split(',') if p.strip()])
                negative_prompt = ', '.join([p.strip() for p in negative_prompt.split(',') if p.strip()])

            width, height = self.get_dimensions()
            current_seed = self.seed if self.seed != 0 else random.randint(1, 2**32 - 1)
            parameters = {
                "steps": self.steps, "sampler": self.sampler, "scale": self.scale, "uc": negative_prompt, 
                "width": width, "height": height, "seed": current_seed, "noise_schedule": self.noise_schedule,
                "smea": self.smea, "smea_dyn": self.smea_dyn, "unclip_guidance_scale": self.unclip_guidance_scale
            }
            payload = {"input": positive_prompt, "model": self.model, "parameters": parameters, "action": "generate"}

            logger.info(f"--- 发送给生图模型的最终Payload ---\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n---------------------------------")
            
            image_data = await self._call_nai_api(payload)
            
            final_image_bytes = image_data
            try:
                with io.BytesIO(image_data) as zip_buffer, zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                    image_filename = next((name for name in zip_ref.namelist() if name.endswith(('.png', '.jpg', '.jpeg'))), None)
                    if image_filename: final_image_bytes = zip_ref.read(image_filename)
            except zipfile.BadZipFile: pass

            save_path = self.save_dir / f"nai_{uuid.uuid4()}.png"
            save_path.write_bytes(final_image_bytes)
            logger.info(f"图片已成功生成并保存至: {save_path}")
            try:
                yield event.chain_result([Image.fromFileSystem(str(save_path))])
            finally:
                if not self.save_images_locally and save_path.exists():
                    save_path.unlink()
                    logger.info(f"临时图片已删除: {save_path}")
        except Exception as e:
            logger.error(f"图片生成任务失败: {e}", exc_info=True)
            yield event.plain_result(f"生成失败: {e}")

    def _create_help_text(self):
        return textwrap.dedent("""\
        NAI Canvas 绘图插件 帮助信息
        ====================================
        核心绘图命令: /nai生图
        用法: /nai生图 [预设名] <内容>
        说明: 如果不指定[预设名]，将自动使用“默认”预设。

        插件会智能区分两种模式：

        一、智能模式 (用于简单/详细描述)
        当您的输入是自然语言时 (如“一个女孩”)，插件会进行创意扩写或翻译，并将结果与预设的【正向】提示词【融合】，同时使用预设的【反向】提示词。
        示例: /nai生图 一个穿白裙的女孩在海边

        二、专业模式 (用于专业提示词/画师串)
        当您的输入是专业提示词格式时，将启用精确的【覆盖】逻辑，规则如下：

        1. 只有正向提示词 (无“|”)
           /nai生图 1girl, masterpiece
           效果: 【覆盖】预设正向，【使用】预设反向。

        2. 正向 | 反向
           /nai生图 1girl | lowres, bad hands
           效果: 【覆盖】预设正向，【覆盖】预设反向。

        3. 正向 | (反向为空)
           /nai生图 1girl, masterpiece |
           效果: 【覆盖】预设正向，【使用】预设反向。

        4. | 反向 (正向为空)
           /nai生图 | lowres, bad hands
           效果: 【使用】预设正向，【覆盖】预设反向。

        ====================================
        提示词管理命令:
        /nai增加提示词 <名称>|<正向>|<反向> (仅机器人所有者)
        /nai删除提示词 <名称> (仅机器人所有者)
        /nai提示词列表
        /nai查看提示词 <名称>
        
        ====================================
        注意事项:
        - LLM增强功能需在插件配置中开启并配置API密钥。
        - 法术解析地址：https://spell.novelai.dev
        """).strip()

    def _get_clean_args(self, full_str: str, command_aliases: list) -> str:
        """获取并清理命令参数，移除命令别名本身。"""
        text = full_str.strip()
        for alias in command_aliases:
            if text.startswith(alias):
                return text[len(alias):].lstrip()
        return text

    @filter.command("nai生图")
    async def handle_nai_sheng_tu(self, event: AstrMessageEvent):
        aliases = ["nai生图"]
        args_str = self._get_clean_args(event.message_str, aliases)

        if not args_str:
            yield event.plain_result("用法: /nai生图 <你的描述或专业提示词>")
            return
        
        try:
            # 1. 预解析用户输入
            has_separator = '|' in args_str
            user_positive_raw = args_str
            user_negative_raw = ""
            if has_separator:
                parts = args_str.split('|', 1)
                user_positive_raw = parts[0].strip()
                user_negative_raw = parts[1].strip()

            # 2. 解析预设名和核心提示词
            preset_name, user_prompt_for_llm = "默认", user_positive_raw
            parts = user_positive_raw.split(maxsplit=1)
            if len(parts) > 1 and parts[0] in self.presets:
                preset_name, user_prompt_for_llm = parts[0], parts[1]
            
            preset = self.presets.get(preset_name, {})
            preset_positive = preset.get("positive", "")
            preset_negative = preset.get("negative", "")

            # 3. 初始化变量
            status_message = ""
            final_positive = ""
            final_negative = ""
            strategy = "增强已禁用" # 默认值

            # 4. LLM分析与逻辑分流
            if self.enable_prompt_enhancement:
                analysis = await self._get_llm_analysis(user_prompt_for_llm)
                strategy = analysis.get("processing_strategy")
                
                # 智能模式：融合逻辑
                if strategy in ['expand', 'translate_and_tagify']:
                    if strategy == 'expand':
                        status_message = "Nai绘图：识别为简单描述，正在创意扩写..."
                        processed_positive = await self._expand_simple_prompt(user_prompt_for_llm)
                    else: # translate_and_tagify
                        status_message = "Nai绘图：识别为详细描述，正在翻译..."
                        processed_positive = await self._translate_chinese_prompt(user_prompt_for_llm)
                    
                    final_positive = f"{preset_positive}, {processed_positive}" if preset_positive else processed_positive
                    final_negative = preset_negative

                # 专业模式：覆盖逻辑
                elif strategy == 'process_directly':
                    status_message = "Nai绘图：识别为专业提示词，正在处理..."
                    
                    # 根据'|'的存在和内容决定正反向提示词
                    if has_separator:
                        if user_positive_raw:
                            final_positive = await self._process_mixed_prompt(user_positive_raw)
                        else: # | text
                            final_positive = preset_positive
                        
                        if user_negative_raw:
                            final_negative = await self._process_mixed_prompt(user_negative_raw)
                        else: # text |
                            final_negative = preset_negative
                    else: # No '|'
                        final_positive = await self._process_mixed_prompt(user_positive_raw)
                        final_negative = preset_negative
                
                else:
                    raise ValueError(f"LLM返回了未知的处理策略: {strategy}")
            else: # LLM增强关闭，默认按专业模式的覆盖逻辑处理
                status_message = "Nai绘图：正在处理您的请求..."
                if has_separator:
                    if user_positive_raw: final_positive = await self._process_mixed_prompt(user_positive_raw)
                    else: final_positive = preset_positive
                    if user_negative_raw: final_negative = await self._process_mixed_prompt(user_negative_raw)
                    else: final_negative = preset_negative
                else:
                    final_positive = await self._process_mixed_prompt(user_positive_raw)
                    final_negative = preset_negative


            # 5. 发送状态消息并执行绘图
            yield event.plain_result(status_message)

            log_details = f"""
            --- 提示词处理详情 ---
            使用预设: {preset_name}
            用户输入 (原始): {args_str}
            LLM处理策略: {strategy}
            最终正向提示词: {final_positive}
            最终反向提示词: {final_negative}
            --------------------------
            """
            logger.info(textwrap.dedent(log_details))

            async for res in self._generate_image_task(event, final_positive, final_negative):
                yield res

        except Exception as e:
            logger.error(f"/nai生图 命令处理失败: {e}", exc_info=True)
            yield event.plain_result(f"处理失败: {e}")

    @filter.command("nai帮助")
    async def handle_nai_help(self, event: AstrMessageEvent):
        try:
            yield event.plain_result(self._create_help_text())
        except Exception as e:
            logger.error(f"生成帮助文本时出错: {e}", exc_info=True)
            yield event.plain_result("生成帮助文本时遇到错误。")

    @filter.command("nai增加提示词")
    async def handle_nai_add_preset(self, event: AstrMessageEvent):
        if not self.is_admin(event):
            return
        
        aliases = ["nai增加提示词"]
        args_str = self._get_clean_args(event.message_str, aliases)
        try:
            parts = args_str.split('|', 2)
            
            if len(parts) < 2: raise ValueError("格式错误")
            
            name = parts[0].strip()
            if not name or name == "默认":
                yield event.plain_result("错误：不能使用'默认'作为名称或名称不能为空。"); return
            
            positive = parts[1]
            negative = parts[2] if len(parts) > 2 else ""

            self.presets[name] = {"positive": positive, "negative": negative}
            if self._save_presets(self.presets):
                yield event.plain_result(f"成功添加/更新提示词: '{name}'")
            else: yield event.plain_result("保存提示词文件失败。")
        except ValueError:
            yield event.plain_result("格式错误！用法: /nai增加提示词 <名称>|<正向>|<反向>")
        except Exception as e:
            yield event.plain_result(f"添加失败: {e}")

    @filter.command("nai删除提示词")
    async def handle_nai_delete_preset(self, event: AstrMessageEvent):
        if not self.is_admin(event):
            return

        aliases = ["nai删除提示词"]
        name = self._get_clean_args(event.message_str, aliases)
        if not name: yield event.plain_result("请输入要删除的提示词名称。"); return
        if name == "默认": yield event.plain_result("错误：不能删除'默认'提示词。"); return
        if name in self.presets:
            del self.presets[name]
            if self._save_presets(self.presets):
                yield event.plain_result(f"成功删除提示词: '{name}'")
            else: yield event.plain_result("保存提示词文件失败。")
        else: yield event.plain_result(f"未找到名为 '{name}' 的提示词。")

    @filter.command("nai提示词列表")
    async def handle_nai_list_presets(self, event: AstrMessageEvent):
        if not self.presets:
            yield event.plain_result("当前没有可用的提示词。"); return
        names = list(self.presets.keys())
        message = "可用提示词列表\n\n- " + "\n- ".join(names)
        yield event.plain_result(message)

    @filter.command("nai查看提示词")
    async def handle_nai_view_preset(self, event: AstrMessageEvent):
        aliases = ["nai查看提示词"]
        name = self._get_clean_args(event.message_str, aliases)
        if not name: yield event.plain_result("请输入要查看的提示词名称。"); return
        preset = self.presets.get(name)
        if preset:
            pos = preset.get('positive') or "(空)"
            neg = preset.get('negative') or "(空)"
            message = f"提示词详情: {name}\n\n正向: {pos}\n\n反向: {neg}"
            yield event.plain_result(message)
        else: yield event.plain_result(f"未找到名为 '{name}' 的提示词。")

    async def terminate(self):
        logger.info("NovelAI 智能绘图插件 已成功停用")
