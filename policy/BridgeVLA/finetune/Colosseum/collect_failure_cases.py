import os
import pickle
import copy
import numpy as np
from PIL import Image

def save_failed_episode(transitions, task_name, variation, episode_id, descriptions, base_dir="/data/colosseum_dataset/failed_eval_demos"):
    task_var_folder = f"{task_name}_{variation}"
    ep_dir = os.path.join(base_dir, task_name, task_var_folder, "variation0", "episodes", f"episode{episode_id}")
    os.makedirs(ep_dir, exist_ok=True)
    
    cameras = ['front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist']
    for cam in cameras:
        os.makedirs(os.path.join(ep_dir, f"{cam}_rgb"), exist_ok=True)
        os.makedirs(os.path.join(ep_dir, f"{cam}_depth"), exist_ok=True)
        
    low_dim_obs_list = []
    actions_list = []
    
    for step_idx, pure_obs in enumerate(transitions):
        try:
            obs_copy = copy.deepcopy(pure_obs)
            actions_list.append(pure_obs.action)

            for cam in cameras:
                rgb_attr = f"{cam}_rgb"
                depth_attr = f"{cam}_depth"
                
                rgb_img = getattr(pure_obs, rgb_attr, None)
                depth_img = getattr(pure_obs, depth_attr, None)
                
                if rgb_img is not None:
                    if rgb_img.ndim == 3 and rgb_img.shape[0] in [1, 3, 4]:
                        rgb_img = np.transpose(rgb_img, (1, 2, 0))
                    if rgb_img.dtype in [np.float32, np.float64]:
                        rgb_img = (rgb_img * 255).astype(np.uint8)
                    Image.fromarray(rgb_img).save(os.path.join(ep_dir, rgb_attr, f"{step_idx}.png"))
                
                if depth_img is not None:
                    depth_img = np.squeeze(depth_img)
                    if depth_img.dtype in [float, np.float32, np.float64]:
                        depth_img = (depth_img * 1000).astype(np.uint16)  
                    Image.fromarray(depth_img).save(os.path.join(ep_dir, depth_attr, f"{step_idx}.png"))
                
                setattr(obs_copy, rgb_attr, None)
                setattr(obs_copy, depth_attr, None)
                setattr(obs_copy, f"{cam}_mask", None)
                setattr(obs_copy, f"{cam}_point_cloud", None)
                    
            low_dim_obs_list.append(obs_copy)
            
        except Exception as e:
            print(f"[Frame {step_idx}] failed: {e}")

    try:
        with open(os.path.join(ep_dir, "low_dim_obs.pkl"), "wb") as f:
            pickle.dump(low_dim_obs_list, f)
        with open(os.path.join(ep_dir, "variation_number.pkl"), "wb") as f:
            pickle.dump(variation, f)
        with open(os.path.join(ep_dir, "variation_descriptions.pkl"), "wb") as f:
            pickle.dump(descriptions, f)
        np.save(os.path.join(ep_dir, "actions.npy"), np.array(actions_list))
        print(f" [Failure Captured] {len(low_dim_obs_list)} frames written into: {ep_dir}")
    except Exception as e:
        print(f"Writing failure: {e}")