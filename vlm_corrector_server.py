import re
import os
import json
import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from peft import PeftModel
from safetensors.torch import load_file
from flask import Flask, request, jsonify 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

app = Flask(__name__)

class VLMCorrector:
    def __init__(self, model_path, base_model_path="/home/zxn/MomaGraph-R1", device="cuda:0"):
        print(f"正在加载纠错 VLM : {model_path} ...")
        self.device = device
 
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            base_model_path, 
            trust_remote_code=True
        )
 
        print("加载模型底座结构...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,           # 先用基础模型结构
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  
        )
 
        # ==========================================
        # 🌟 Smart Weight Loading: SFT vs RL LoRA
        # ==========================================
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            # Branch A: PEFT LoRA Weights (RL tuned)
            print("📦 Detected LoRA Adapter. Mounting PEFT weights...")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            
            # 🚀 CRITICAL SPEEDUP: Merge LoRA weights into base model
            print("⚡ Merging LoRA weights into base model for max inference speed...")
            self.model = self.model.merge_and_unload()
            
            print("✅ LoRA weights successfully merged!")
            
        else:
            # Branch B: Standard SFT Full Weights
            print("🔍 No LoRA format detected. Using standard full-weight loading...")
            checkpoint_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(checkpoint_path):
                state_dict = load_file(checkpoint_path)
            else:
                checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
                state_dict = torch.load(checkpoint_path, map_location="cpu")

            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = self._remap_key(k)
                new_state_dict[new_key] = v

            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")
        # ==========================================

        self.model.eval()
        self.history_messages = [] # 初始化为空列表
        print("VLM 服务端初始化成功，等待接收请求...")

    @staticmethod
    def _remap_key(k): 
        # 1. 独立处理 lm_head
        if "lm_head" in k:
            return "lm_head.weight" if "weight" in k else "lm_head.bias"
        # 2. 视觉模块 (Visual Encoder)
        if "visual." in k:
            # 以 "visual." 为界截断，保留后面的完整结构
            return "model.visual." + k.split("visual.", 1)[1]
        # 3. 语言模型主干 (Layers, Embeddings, Norm)
        for keyword in ["layers.", "embed_tokens.", "norm."]:
            if keyword in k:
                return "model.language_model." + keyword + k.split(keyword, 1)[1]
        return k 

    def _convert_to_pil(self, img_array):
        """将 numpy 数组转换为 PIL Image"""
        if isinstance(img_array, Image.Image):
            return img_array
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            img_array = (img_array * 255).astype(np.uint8)
        if len(img_array.shape) == 3 and img_array.shape[0] in [1, 3, 4]:
            img_array = np.transpose(img_array, (1, 2, 0))
        return Image.fromarray(img_array)

    def _convert_to_qwen_messages(self, prompt_messages):
        """ 对齐训练集的 Block 转换逻辑 """
        converted = []
        for msg in prompt_messages:
            role, content = msg["role"], msg["content"]
            images = msg.get("images", [])
            if role == "user" and images:
                blocks = [{"type": "image", "image": img} for img in images]
                blocks.append({"type": "text", "text": content})
                converted.append({"role": role, "content": blocks})
            else:
                converted.append({"role": role, "content": content})
        return converted

    def _get_current_text(self, instruction, base_proposal_list, time_step):
        """ 严格对齐训练集的 Prompt 字符串 """
        if time_step == 0:
            return (
                f"Task instruction: {instruction}\n"
                f"[Time Step: 0] Task initiation. Analyze the multi-view images to extract the initial scene graph. "
                f"Evaluate the initial task progress, and predict the resulting spatial changes if the Action Proposal {base_proposal_list} is applied."
            )
        else:
            return (
                f"[Time Step: {time_step}] Previous action executed. Given the new observations and the Action Proposal: {base_proposal_list}, "
                f"perform a structured physical dual-check:\n"
                f"1. Backward check: Compare History vs. Current to quantify geometric distance changes and identify any physical stasis or unintended deviations.\n"
                f"2. Forward check: Predict how the Current Scene Graph (specifically edge distances and spatial relations) and task progress will change if the Proposal is executed.\n"
                f"Synthesize these checks to objectively determine if a correction is needed. Output the structured evaluation and the corrected action."
            )
    # def _get_current_text(self, instruction, base_proposal_list, time_step, num_images=4):
    #     """生成当前时刻的用户文本（包含图像占位符）""" 
    #     image_tags = "<image>\n" * num_images
        
    #     if time_step == 0:
    #         return (
    #             f"{image_tags}Task instruction: {instruction}\n"
    #             f"[Time 0] Task initiation. Analyze the multi-view images to extract the initial scene graph. "
    #             f"Predict the future state if the Action Proposal {base_proposal_list} is applied, and evaluate the initialization."
    #         )
    #     else:
    #         return (
    #             f"{image_tags}[Time t] Previous action executed. Given the new observations and the Action Proposal: {base_proposal_list}, "
    #             f"perform a physical dual-check:\n1. Backward check (History vs. Current)\n2. Forward check (Current vs. Future Prediction)\n"
    #             f"Output the structured evaluation and the corrected action."
    #         )

    def reset_memory(self):
        self.history_messages = []

    def step(self, current_images, base_policy_proposal, time_step, instruction):
        """
        执行一步推理，返回纠正后的动作。
        Args:
            current_images: list of numpy arrays (C, H, W) 或 (H, W, C)
            base_policy_proposal: 基础策略输出的动作（列表长度 <= 8）
            time_step: 当前时间步（0 或 >0）
            instruction: 任务指令
        Returns:
            corrected_action: 纠正后的动作（列表）
        """

        base_proposal_list = [float(v) for v in base_policy_proposal[:8]]
        current_text = self._get_current_text(instruction, base_proposal_list, time_step)

        # =====================================================================
        #  DEBUG 1: 检查客户端传来的物理图片格式
        # =====================================================================
        if time_step in [0, 1]:
            print("\n" + "▼"*60)
            print(f"🚦 [EVAL 对齐核查] t={time_step} 样本全链路数据流拆解")
            print("▼"*60)
            img_types = [type(img).__name__ for img in current_images]
            print(f"▶ [1. 物理图片]: 成功接收了 {len(current_images)} 张图片, 格式分布: {img_types}")

        # 1. 拼接历史记忆（纯文本）与当前观察（带图像）
        raw_messages = self.history_messages + [{"role": "user", "content": current_text, "images": current_images}]

        # 2. 推理
        qwen_msgs = self._convert_to_qwen_messages(raw_messages)

        text = self.processor.apply_chat_template(
            qwen_msgs, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(qwen_msgs)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        # =====================================================================
        #  DEBUG 2: 打印送入模型的真实 Prompt 结构 (Token 级别)
        # =====================================================================
        if time_step in [0, 1]:
            pixel_shape = inputs.get('pixel_values', torch.empty(0)).shape
            print(f"\n▶ [2. Pixel Values Shape]: {pixel_shape}")
            print(f"\n▶ [3. 模型底层的真实 Input Tokens (重点核对 <|vision_start|> 位置)]:")
            # 解码 input_ids，保留所有特殊 token 以供核对
            # decoded_prompt = self.processor.decode(inputs.input_ids[0], skip_special_tokens=False)
            # print(decoded_prompt)
            print("▲"*60 + "\n")


        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        response_text = self.processor.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
 
        # 3. 核心对齐：重构记忆。提取 JSON，构造下一帧要看的纯文本历史 
        # 提取场景图和任务进度
        graph_match = re.search(r'Current Scene Graph \(t=\d+\):\s*(\{.*?\})\n', response_text)
        prog_match = re.search(r'Global Task Progress:\s*([0-9.]+)', response_text)
        
        # scene_graph = json.loads(graph_match.group(1)) if graph_match else {}
        # 🌟 替换掉原来的 scene_graph = json.loads(...)
        scene_graph = {}
        if graph_match:
            graph_str = graph_match.group(1)
            try:
                # 尝试标准的严格 JSON 解析
                scene_graph = json.loads(graph_str)
            except Exception:
                try:
                    # 挽救大模型幻觉：兼容单引号和 Python 字典格式
                    import ast
                    scene_graph = ast.literal_eval(graph_str)
                except Exception as e:
                    print(f"⚠️ [防弹警告] 场景图解析彻底失败，已安全回退为空字典。异常文本: {graph_str}")
                    scene_graph = {}
        progress = float(prog_match.group(1)) if prog_match else 0.0
        
        # 构造符合训练集 T_PREV 格式的 JSON
        history_json_dict = {
            "scene_graph": scene_graph,
            "progress": progress
        }
        clean_json = f"```json\n{json.dumps(history_json_dict, indent=2)}\n```"
        
        # 严格对齐训练集格式
        past_user_text = f"Task instruction: {instruction}\n[Time Step: {time_step}] Analyze the historical observations to extract the past scene graph and task progress."
        past_assistant_text = f"{clean_json}\nAction Executed: {base_proposal_list}"

        # 滑动窗口：清空，只存入这一帧转换后的文本记忆
        self.history_messages = [
            {"role": "user", "content": past_user_text},
            {"role": "assistant", "content": past_assistant_text}
        ]

        return base_proposal_list, response_text

    def _parse_action(self, response_text, fallback_action):
        """从响应中提取 JSON 格式的 corrected_action"""
        try:
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                result_dict = json.loads(json_match.group(1))
                if "corrected_action" in result_dict:
                    return result_dict["corrected_action"][:8]
        except Exception:
            pass
        return fallback_action
 

# ==========================================
# Flask Web API 接口层
# ==========================================

# # 全局初始化
# model_path = "/data/colosseum_dataset/checkpoints/sft/v1/checkpoint-epoch-1"
# base_model_path = "/home/zxn/MomaGraph-R1"
# engine = VLMCorrector(model_path=model_path, device="cuda:4")

@app.route('/step', methods=['POST'])
def handle_step():
    data = request.json
    pil_images = [Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB") for img_b64 in data['images_b64']]
    _, cot_text = engine.step(pil_images, data['base_policy_proposal'], data['time_step'], data['instruction'])
    
    # 提取 corrected_action 返回给客户端
    corrected_action = data['base_policy_proposal']
    correction_needed = False  # 🌟 新增默认值
    try:
        json_match = re.search(r'```json\n(.*?)\n```', cot_text, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group(1))
            corrected_action = parsed_json.get("corrected_action", corrected_action)[:8]
            correction_needed = parsed_json.get("correction_needed", False)
    except: pass

    # return jsonify({"corrected_action": corrected_action, "cot": cot_text})
    return jsonify({
        "corrected_action": corrected_action, 
        "cot": cot_text, 
        "correction_needed": correction_needed
    })


@app.route('/reset', methods=['POST'])
def handle_reset():
    engine.reset_memory()
    return jsonify({"status": "success"})

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to VLM checkpoint")
    parser.add_argument("--port", type=int, default=5050) 
    parser.add_argument("--gpu", type=str, default="cuda:0")
    args = parser.parse_args()

    engine = VLMCorrector(model_path=args.ckpt, device=args.gpu)

    # 启动本地服务，端口 5050
    print(f"启动 Flask 服务端，监听端口 {args.port}...")
    app.run(host='0.0.0.0', port=args.port, threaded=False)




# --- Server command: conda activate cot_sft ---
# nohup python3 vlm_corrector_server.py \
    # --ckpt /data/colosseum_dataset/checkpoints/sft/v1/checkpoint-epoch-1 \
    # --gpu cuda:4 \
    # --port 5050 > logs/vlm_server_ep1_5050.log 2>&1 &

# nohup python3 vlm_corrector_server.py \
#     --ckpt /data/colosseum_dataset/easyr1_ckpt/checkpoints/easy_r1/momagraph_dapo_multitask_baseline/epoch_archive/epoch_1_step_110/actor/lora_adapter \
#     --gpu cuda:3 \
#     --port 5054 > logs/vlm_server_lora_5054.log 2>&1 &


# --- Evaluation command: conda activate bridgevla ---
# CUDA_VISIBLE_DEVICES=4 VLM_PORT=5050 nohup bash /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum/eval_worker.sh \
#     "ep1_resume_test" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 4 \
#     "setup_chess scoop_with_spatula stack_cups" > /home/zxn/Chain-of-Correction/logs/eval_ep1_remaining.log 2>&1 &
# "put_money_in_safe place_wine_at_rack_location meat_on_grill move_hanger open_drawer turn_oven_on setup_chess scoop_with_spatula stack_cups" 