import re
import os
import json
import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO


class VLMCorrector:
    # 接收 eval.py 传来的所有参数，但统统忽略（因为真正的模型在 server 端）
    def __init__(self, *args, **kwargs):
        # self.server_url = "http://127.0.0.1:5050"
        vlm_port = os.environ.get("VLM_PORT", "5050")
        self.server_url = f"http://127.0.0.1:{vlm_port}"
        
        # 频率锁：避免每帧都做网络请求导致环境卡顿。现在设定为每 5 帧干预一次。
        self.check_freq = 1 
        
        print(f"📡 [Client] VLM 客户端已启动！") 
        
        # 尝试联通测试并重置服务端记忆
        try:
            requests.post(f"{self.server_url}/reset", timeout=5)
            print(f"✅ [Client] 已联通 Server: {self.server_url}")
        except Exception as e:
            print(f"⚠️ [Client] 无法连接到 VLM 服务器，请检查服务端是否运行。")

    def reset_memory(self):
        """通知服务端清空上一集的对话记忆"""
        try:
            requests.post(f"{self.server_url}/reset", timeout=5)
        except Exception:
            pass

    def _array_to_b64(self, img_array):
        """将仿真器传来的 Numpy 数组转为 Base64 字符串，使用 JPEG 极致压缩提速"""
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            img_array = (img_array * 255).astype(np.uint8)
        if len(img_array.shape) == 3 and img_array.shape[0] in [1, 3, 4]:
            img_array = np.transpose(img_array, (1, 2, 0))
            
        pil_img = Image.fromarray(img_array)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=85) 
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def step(self, current_images, base_policy_proposal, time_step, instruction):
        # 统一转为 float64 列表以便 JSON 序列化
        base_proposal_list = [round(float(v), 6) for v in base_policy_proposal]
        
        # 频率锁生效：如果不是整数倍帧，直接丝滑返回基础动作，不打扰物理仿真
        if time_step != 0 and time_step % self.check_freq != 0:
            return base_proposal_list, False

        # 到了触发干预的帧，打包图片和数据发送给 Server
        images_b64 = [self._array_to_b64(img) for img in current_images]
        
        payload = {
            "images_b64": images_b64,
            "base_policy_proposal": base_proposal_list,
            "time_step": time_step,
            "instruction": instruction
        }
        
        try:
            # 向独立运行的大脑发送请求
            response = requests.post(f"{self.server_url}/step", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # 打印服务端传回来的 CoT 思维链（可选）
            # cot = result.get('cot', '')
            # if cot:
            #     print(f"\n🧠 [VLM 思维链 t={time_step}] \n{cot}\n")
            corrected_action = result.get("corrected_action", base_proposal_list)[:8]
 
            # # 判定是否真正干预
            # # 逻辑：如果返回的动作和原动作不同，或者服务端明确标记了需要纠错
            # # 这里依赖服务端返回的推理结果
            # is_correction_needed = False
            # try:
            #     # 再次解析 CoT 以提取 correction_needed 标志位
            #     json_match = re.search(r'```json\n(.*?)\n```', cot, re.DOTALL)
            #     if json_match:
            #         parsed = json.loads(json_match.group(1))
            #         is_correction_needed = parsed.get("correction_needed", False)
            # except: pass
            is_correction_needed = result.get("correction_needed", False)
             
            return corrected_action, is_correction_needed
            
        except Exception as e:
            print(f"❌ [Client] VLM 服务调用失败: {e}")
            return base_proposal_list, False