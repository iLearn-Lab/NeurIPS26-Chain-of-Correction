import re
import os
import json
import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO


class VLMCorrector:
    """
    Client-side interface for the Vision-Language Model (VLM) correction server.
    This class handles image encoding and network communication with a remote 
    VLM server that performs action correction via Chain-of-Thought (CoT).
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the VLM Client.
        
        Args:
            check_freq (int): Frequency of interventions (e.g., 5 means query every 5 steps).
        """
        vlm_port = os.environ.get("VLM_PORT", "5050")
        self.server_url = f"http://127.0.0.1:{vlm_port}"
        
        # Frequency lock: avoids network overhead by limiting server queries.
        self.check_freq = 1 
        
        print(f"[VLM Client] Initializing...") 
        
        # Connection test and initial memory reset
        try:
            requests.post(f"{self.server_url}/reset", timeout=5)
            print(f"[VLM Client] Connected to server at: {self.server_url}")
        except Exception:
            print(f"[VLM Client] Warning: Unable to reach VLM server. Ensure the backend is running.")

    def reset_memory(self):
        """Notifies the server to clear conversation history for a new episode."""
        try:
            requests.post(f"{self.server_url}/reset", timeout=5)
        except Exception:
            pass

    def _array_to_b64(self, img_array):
        """
        Converts a NumPy image array into a Base64 string with JPEG compression.
        
        Args:
            img_array (np.ndarray): Image array from the simulator.
            
        Returns:
            str: Base64 encoded JPEG string.
        """
        # Convert float normalized images (0-1) to uint8 (0-255)
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            img_array = (img_array * 255).astype(np.uint8)
        if len(img_array.shape) == 3 and img_array.shape[0] in [1, 3, 4]:
            img_array = np.transpose(img_array, (1, 2, 0))
            
        pil_img = Image.fromarray(img_array)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=85) 
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def step(self, current_images, base_policy_proposal, time_step, instruction):
        """
        Executes a single environment step. Queries the VLM if the frequency lock allows.
        
        Args:
            current_images (list): List of camera observations (NumPy arrays).
            base_policy_proposal (np.ndarray): The raw action proposed by the base policy.
            time_step (int): Current episode step index.
            instruction (str): Natural language task instruction.
            
        Returns:
            tuple: (action_to_execute, is_correction_needed)
        """
        # Normalize to list of floats for JSON compatibility
        base_proposal_list = [round(float(v), 6) for v in base_policy_proposal]
        
        # Frequency lock: Return base action immediately if not an intervention frame
        if time_step != 0 and time_step % self.check_freq != 0:
            return base_proposal_list, False

        # Prepare multimodal payload
        images_b64 = [self._array_to_b64(img) for img in current_images]
        
        payload = {
            "images_b64": images_b64,
            "base_policy_proposal": base_proposal_list,
            "time_step": time_step,
            "instruction": instruction
        }
        
        try:
            # POST request to the VLM reasoning engine
            response = requests.post(f"{self.server_url}/step", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Extract correction results
            corrected_action = result.get("corrected_action", base_proposal_list)[:8]

            # is_correction_needed = False
            # try:
            #     json_match = re.search(r'```json\n(.*?)\n```', cot, re.DOTALL)
            #     if json_match:
            #         parsed = json.loads(json_match.group(1))
            #         is_correction_needed = parsed.get("correction_needed", False)
            # except: pass
            is_correction_needed = result.get("correction_needed", False)
             
            return corrected_action, is_correction_needed
            
        except Exception as e:
            print(f"[VLM Client] Error calling VLM server: {e}")
            return base_proposal_list, False