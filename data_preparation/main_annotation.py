import os
import cv2
import json
import pickle
import numpy as np
import rlbench.tasks
from pyrep.const import JointType
from pyrep.objects.shape import Shape
from pyrep.errors import WrongObjectTypeError
from rlbench.environment import Environment
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
import colosseum.rlbench.tasks

from progress_estimator import ProgressEstimator
from extract_momagraph import extract_momagraph_frame
from utils import draw_wrapped_text, EndEffectorPoseViaPlanning2, get_graph_diff


# ==============================================================================
# Global Task Registry
# Centralized management for task-specific object names and state vector slices.
# mo: Manipulated Object | to: Target Object
# ==============================================================================
TASK_REGISTRY = {
    "basketball_in_hoop": {
        "objects": {
            "mo": "ball", 
            "to": "basket_ball_hoop_respondable"
        },
        "semantic_mo_name": "ball",
        "semantic_to_name": "hoop",
        "indices": {
            "ball": slice(21, 28),
            "hoop_respondable": slice(7, 14),
            "success": slice(35, 42)
        }
    },
    "close_box": {
        "objects": {
            "mo": "box_lid", 
            "to": "box_base"
        },
        "semantic_mo_name": "box_lid",
        "semantic_to_name": "box_base",
        "indices": {
            "box_base": slice(0, 7),
            "box_joint": slice(14, 21),
            "box_joint_angle": 22,
            "box_lid": slice(36, 43)
        }
    },
    "close_laptop_lid": {
        "objects": {
            "mo": "lid", 
            "to": "base"
        },
        "semantic_mo_name": "laptop_lid",
        "semantic_to_name": "laptop_base",
        "indices": {
            "base": slice(0, 7),
            "lid": slice(36, 43) 
        }
    },
    "empty_dishwasher": {
        "objects": {
            "door": "dishwasher_door",
            "tray": "dishwasher_tray",
            "plate": "dishwasher_plate",
            "dishwasher": "dishwasher"
        },
        "semantic_mo_name": "plate",
        "semantic_to_name": "dishwasher_tray",
        "indices": {
            "dishwasher": slice(7, 14),
            "door": slice(93, 100),  
            "tray": slice(100, 107),
            "plate": slice(121, 128),
        }
    },
    "get_ice_from_fridge": {
        "objects": {
            "cup": "cup",
            "fridge": "fridge_base",
            "tongue": "tongue"
        },
        "semantic_mo_name": "cup",
        "semantic_to_name": "tongue",
        "indices": {
            "fridge": slice(7, 14),      # fridge_base
            "cup": slice(14, 21),        # cup
            "joint": slice(21, 28),      # joint 
            "tongue": slice(71, 78)      # tongue
        }
    },
    "hockey": {
        "objects": {
            "stick": "hockey_stick",
            "ball": "hockey_ball",
            "goal": "hockey_goal"
        },
        "semantic_mo_name": "stick", 
        "semantic_to_name": "goal",
        "indices": {
            "goal": slice(0, 7),     # hockey_goal
            "stick": slice(21, 28),  # hockey_stick  
            "ball": slice(28, 35)    # hockey_ball  
        }
    },
    "insert_onto_square_peg": {
        "objects": {
            "ring": "square_ring",
            "pillar0": "pillar0",
            "pillar1": "pillar1",
            "pillar2": "pillar2"
        },
        "semantic_mo_name": "square_ring", 
        "semantic_to_name": "target_pillar",
        "indices": {
            "pillar0": slice(7, 14),
            "pillar1": slice(14, 21),
            "pillar2": slice(21, 28),
            "ring": slice(56, 63),       # square_ring
            "success_centre": slice(63, 70) 
        }
    },
    "meat_on_grill": {
        "objects": {
            "chicken": "chicken",
            "steak": "steak",
            "grill": "grill"
        },
        "semantic_mo_name": "target_meat", 
        "semantic_to_name": "grill",
        "indices": {
            "grill": slice(0, 7),        # grill
            "chicken": slice(14, 21),    # chicken  
            "success": slice(35, 42),
            "steak": slice(42, 49)       # steak  
        }
    },
    "move_hanger": {
        "objects": {
            "hanger": "clothes_hanger0",
            "rack1": "clothes_rack",
            "rack2": "clothes_rack0"
        },
        "semantic_mo_name": "hanger", 
        "semantic_to_name": "target_rack",
        "indices": {
            "rack1": slice(0, 7),     # clothes_rack
            "rack2": slice(7, 14),    # clothes_rack0
            "hanger": slice(56, 63)   # clothes_hanger0 
        }
    },
    "open_drawer": {
        "objects": {
            "bottom": "drawer_bottom",
            "middle": "drawer_middle",
            "top": "drawer_top"
        },
        "semantic_mo_name": "drawer", 
        "semantic_to_name": "opened_drawer",        
        "indices": {           
            "bottom": slice(45, 52),               
            "middle": slice(66, 73),               
            "top": slice(73, 80)                   
        }
    },
    "place_wine_at_rack_location": {
        "objects": {
            "wine": "wine_bottle",
            "rack": "rack_bottom"
        },
        "semantic_mo_name": "wine_bottle",
        "semantic_to_name": "rack_section",
        "indices": {
            "wine": slice(28, 35),           # wine_bottle
            "rack": slice(35, 42),           # rack_bottom
            "target_middle": slice(7, 14),   # success_middle 
            "target_left": slice(14, 21),    # success_left   
            "target_right": slice(21, 28)    # success_right 
        }
    },
    "put_money_in_safe": {
        "objects": {
            "money": "dollar_stack",
            "safe": "safe_body"
        },
        "semantic_mo_name": "money",
        "semantic_to_name": "safe_shelf", 
        "indices": {
            "money": slice(50, 57),           # dollar_stack 
            "safe": slice(0, 7),              # safe_body
            "target_bottom": slice(85, 92),   # success_detector0 (bottom)
            "target_middle": slice(78, 85),   # success_detector1 (middle)
            "target_top": slice(71, 78)       # success_detector2 (top)
        }
    },
    "reach_and_drag": {
        "objects": {
            "stick": "stick",
            "cube": "cube",
            "target": "target0"
        },
        "semantic_mo_name": "stick", 
        "semantic_to_name": "target_zone", 
        "indices": {
            "cube": slice(0, 7),                # cube
            "stick": slice(21, 28),             # stick (3 * 7 = 21)
            "target": slice(28, 35),            # target0
            "success": slice(84, 91)            # success0 (12 * 7 = 84)
        }
    },
    "scoop_with_spatula": {
        "objects": {
            "spatula": "scoop_with_spatula_spatula",
            "cuboid": "Cuboid" 
        },
        "semantic_mo_name": "spatula", 
        "semantic_to_name": "target_zone", 
        "indices": {
            "spatula": slice(0, 7),      # scoop_with_spatula_spatula
            "cuboid": slice(7, 14),      # Cuboid
            "success": slice(28, 35)     # success_sensor (4 * 7 = 28)
        }
    },
    "setup_chess": {
        "objects": {
            "chess_piece": "white_pawn_a_dynamic" 
        }, 
        "semantic_mo_name": "chess_piece",
        "semantic_to_name": "target_zone",
        "indices": {
            "detectors_slice": slice(84, 308), 
            "pieces_slice": slice(406, 630)    
        }
    },
    "slide_block_to_target": {
        "objects": {
            "mo": "block", 
            "to": "target"
        },
        "semantic_mo_name": "block",
        "semantic_to_name": "target_zone",
        "indices": {
            "target": slice(0, 7),
            "block": slice(7, 14),
            "success": slice(14, 21)  
        }
    },
    "stack_cups": {
        "objects": {
            "cup1": "cup1",
            "cup2": "cup2",
            "cup3": "cup3"
        },
        "semantic_mo_name": "active_cup",
        "semantic_to_name": "target_zone",
        "indices": {
            "cup1": slice(0, 7),         # cup1
            "cup2": slice(7, 14),        # cup2 (Base cup)
            "cup3": slice(14, 21),       # cup3
            "success": slice(63, 70)     # success_sensor (9 * 7 = 63)
        }
    },
    "straighten_rope": {
        "objects": {
            "head": "head", 
            "tail": "tail"
        },
        "semantic_mo_name": "rope_end",      
        "semantic_to_name": "target_patch", 
        "indices": {
            "tail": slice(0, 7),             # tail
            "success_tail": slice(28, 35),   # success_tail 
            "success_head": slice(35, 42),   # success_head 
            "head": slice(325, 332)          # head  
        }
    },
    "turn_oven_on": {
        "objects": {
            "knob": "oven_knob",
            "oven": "oven_frame"
        },
        "semantic_mo_name": "oven_knob",
        "semantic_to_name": "oven",
        "indices": {
            "oven": slice(14, 21),       # oven_frame
            "knob": slice(115, 122),     # oven_knob  
            "knob_joint": 99             # oven_knob_joint
        }
    },
    "wipe_desk": {
        "objects": {
            "sponge": "sponge",
            "boundary": "dirt_boundary"
        },
        "semantic_mo_name": "sponge",
        "semantic_to_name": "desk_surface",
        "indices": {
            "sponge": slice(0, 7),           # sponge
            "dirt_boundary": slice(7, 14),   # dirt_boundary
            "sponge_sensor": slice(21, 28),  # sponge_sensor 
        }
    }
}


 
def render_trajectory_video(task_name, episode_dir, config, output_name="momagraph.mp4"):
    """
    Decoupled video renderer that generates a visualization of the scene graph and progress.
    """
    # Load low-dimensional observations
    obs_file = os.path.join(episode_dir, "low_dim_obs.pkl")
    if not os.path.exists(obs_file):
        print(f"[Error] Observation file not found: {obs_file}")
        return
        
    with open(obs_file, "rb") as f:
        data_demo = pickle.load(f)
        
    img_dir = os.path.join(episode_dir, "left_shoulder_rgb")
    if not os.path.exists(img_dir):
        print(f"[Error] RGB image directory not found: {img_dir}")
        return
        
    # Initialize visualization parameters
    num_frames = len(data_demo)
    
    first_img = cv2.imread(os.path.join(img_dir, "0.png"))
    orig_h, orig_w, _ = first_img.shape
    
    render_w = max(640, orig_w) 
    render_h = int(orig_h * (render_w / orig_w))
    
    margin_x = int(render_w * 0.03)  
    margin_y = int(render_h * 0.03)  
    panel_h  = int(render_h * 0.32)  
    
    font_scale_main = render_h / 800.0  
    font_scale_sub  = render_h / 1200.0 
    thick_main = max(1, int(render_h / 300))
    thick_sub  = max(1, int(render_h / 400))
    
    line1_y = int(panel_h * 0.30)
    line2_y = int(panel_h * 0.55)
    
    bar_y1 = int(panel_h * 0.85)
    bar_y2 = int(panel_h * 0.95)
    bar_x1 = int(render_w * 0.05)
    bar_x2 = int(render_w * 0.95)
    bar_max_len = bar_x2 - bar_x1

    # Video Writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    render_fps = 5.0
    out_video = cv2.VideoWriter(output_name, fourcc, render_fps, (render_w, render_h))
    
    print(f"[Status] Rendering video: {orig_w}x{orig_h} -> {render_w}x{render_h}...")
    
    estimator = ProgressEstimator(manipulate_obj=config["semantic_mo_name"], target_obj=config["semantic_to_name"])
    
    # Load task instructions
    inst_file = os.path.join(episode_dir, "variation_descriptions.pkl")
    with open(inst_file, "rb") as f:
        instructions = pickle.load(f)
        
    mock_obs_0 = data_demo[0].__dict__
    prev_obs_dict = mock_obs_0
    font = cv2.FONT_HERSHEY_SIMPLEX

    def _find_edge(edges_list, obj1, obj2):
        for e in edges_list:
            if e.get("object1") == obj1 and e.get("object2") == obj2:
                return e
        return None

    for t in range(num_frames):
        img_path = os.path.join(img_dir, f"{t}.png")
        frame_img = cv2.imread(img_path)
        if frame_img is None: break
        
        frame_img = cv2.resize(frame_img, (render_w, render_h), interpolation=cv2.INTER_CUBIC)
            
        curr_obs_dict = data_demo[t].__dict__
        curr_graph = extract_momagraph_frame(
            task_name=task_name,
            obs_dict=curr_obs_dict, 
            init_obs_dict=mock_obs_0, 
            indices=config["indices"], 
            extents=extents_dict,
            instructions=instructions,
            prev_obs_dict=prev_obs_dict
        ) 
        progress = estimator.estimate(curr_graph)
        action_type = curr_graph["action_type"]
        
        focus_edges = curr_graph.get("focus_edges")
        if not focus_edges:
            mo_name = config.get("semantic_mo_name", "object")
            to_name = config.get("semantic_to_name", "target")
            focus_edges = [("gripper", mo_name), (mo_name, to_name)]
            
        edge_a_tuple = focus_edges[0] if len(focus_edges) > 0 else ("gripper", "entity")
        edge_b_tuple = focus_edges[1] if len(focus_edges) > 1 else None
        
        obj1_a, obj2_a = edge_a_tuple
        edge_a = _find_edge(curr_graph["edges"], obj1_a, obj2_a)
        rels_a = edge_a["spatial_relations"] if edge_a else []
 
        # HUD Rendering
        overlay = frame_img.copy()
        cv2.rectangle(overlay, (margin_x, margin_y), (render_w - margin_x, margin_y + panel_h), (0, 0, 0), -1)
        frame_img = cv2.addWeighted(overlay, 0.65, frame_img, 0.35, 0)
        
        cv2.putText(frame_img, f"Frame: {t:03d} | Intention: {action_type.upper()}", 
                    (margin_x + 10, margin_y + line1_y), font, font_scale_main, (0, 255, 255), thick_main)
        
        max_text_width = render_w - margin_x * 2 - 20 
        
        text_edge1 = f"{obj1_a} -> {obj2_a}: {', '.join(rels_a)}"
        next_y = draw_wrapped_text(frame_img, text_edge1, margin_x + 10, margin_y + line2_y, 
                                   font, font_scale_sub, (255, 255, 255), thick_sub, max_text_width)
        
        if edge_b_tuple:
            obj1_b, obj2_b = edge_b_tuple
            edge_b = _find_edge(curr_graph["edges"], obj1_b, obj2_b)
            rels_b = edge_b["spatial_relations"] if edge_b else []
            text_edge2 = f"{obj1_b} -> {obj2_b}: {', '.join(rels_b)}"
            _ = draw_wrapped_text(frame_img, text_edge2, margin_x + 10, next_y + 5, 
                                font, font_scale_sub, (255, 255, 255), thick_sub, max_text_width)
        
        bar_filled = int(bar_max_len * progress)
        cv2.rectangle(frame_img, (bar_x1, margin_y + bar_y1), (bar_x2, margin_y + bar_y2), (100, 100, 100), -1) 
        cv2.rectangle(frame_img, (bar_x1, margin_y + bar_y1), (bar_x1 + bar_filled, margin_y + bar_y2), (0, 255, 0), -1) 
        
        text = f"Progress: {progress*100:.1f}%"
        text_size = cv2.getTextSize(text, font, font_scale_sub, thick_sub)[0]
        text_x = bar_x1 + (bar_max_len - text_size[0]) // 2
        text_y = margin_y + bar_y1 + (bar_y2 - bar_y1 + text_size[1]) // 2
        cv2.putText(frame_img, text, (text_x, text_y), font, font_scale_sub, (0, 0, 0), thick_sub)
        
        out_video.write(frame_img)
        prev_obs_dict = curr_obs_dict
     
    freeze_frames = int(render_fps * 3)
    for _ in range(freeze_frames):
        out_video.write(frame_img)
 
    out_video.release()
    print(f"[Success] Video saved as: {output_name}")


# ==============================================================================
# Execution Logic
# ==============================================================================
if __name__ == "__main__":
    # Configuration
    task_name = "scoop_with_spatula"    
    # "basketball_in_hoop"      close_box               close_laptop_lid    empty_dishwasher                get_ice_from_fridge     hockey              insert_onto_square_peg
    # "meat_on_grill"           move_hanger             open_drawer         place_wine_at_rack_location     put_money_in_safe       reach_and_drag      scoop_with_spatula
    # setup_chess               slide_block_to_target   stack_cups          straighten_rope                 turn_oven_on            wipe_desk
    demo_number = 95

    config = TASK_REGISTRY[task_name]
    task_class_name = "".join([word.capitalize() for word in task_name.split("_")])
    if "chess" in task_name:
        TaskClass = getattr(colosseum.rlbench.tasks, task_class_name)
    else:
        TaskClass = getattr(rlbench.tasks, task_class_name)

    # --------------------------------------------------------------------------
    # Phase 1: Online BBox and Extent Extraction (Requires PyRep)
    # --------------------------------------------------------------------------
    print(f"[Phase 1] Launching simulator to extract static physical extents...")
    env = Environment(
        action_mode = MoveArmThenGripper(
            arm_action_mode = EndEffectorPoseViaPlanning2(), gripper_action_mode = Discrete()),
        headless=True)
    env.launch()
    rlbench_task = env.get_task(TaskClass)
    rlbench_task.sample_variation() 

    # Extract Gripper Radius
    gripper = rlbench_task._robot.gripper
    gripper_max_width = 0.0
    for joint in gripper.joints:
        if joint.get_joint_type() == JointType.PRISMATIC:
            _, interval = joint.get_joint_interval()
            gripper_max_width += interval[1] 
    if gripper_max_width <= 1e-4:
        gripper_max_width = 0.08  
    gripper_tcp_radius = gripper_max_width / 2.0

    extents_dict = {
        "gripper_radius": gripper_tcp_radius,
        "success_radius": 0.05
    }

    # Extract registered object bounding boxes
    for key, sim_name in config.get("objects", {}).items():
        try:
            bbox = Shape(sim_name).get_bounding_box()
            size = [bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]]
            extents_dict[f"{key}_size"] = np.array(size)
            print(f" |- Extracted {key} size: {extents_dict[f'{key}_size']}")
        except Exception as e:
            print(f" [Warning] Could not extract size for {sim_name}: {e}")

    env.shutdown()
    print("[Phase 1] Static parameters extracted. Simulator closed.")

    # --------------------------------------------------------------------------
    # Phase 2: Offline Trajectory Analysis and Rendering
    # --------------------------------------------------------------------------
    FAILED_EPISODE_DIR = f"/data/colosseum_dataset/failed_eval_demos/{task_name}/{task_name}_0/variation0/episodes/episode20"
    print(f"\n[Phase 2] Parsing demonstration from: {FAILED_EPISODE_DIR}")
    
    # with open(f"/data3/grail/train/{task_name}/all_variations/episodes/episode{demo_number}/variation_descriptions.pkl", "rb") as f:
    #     instructions = pickle.load(f)
    # with open(f"/data3/grail/train/{task_name}/all_variations/episodes/episode{demo_number}/low_dim_obs.pkl","rb") as f:
    #     data_demo = pickle.load(f)

    with open(os.path.join(FAILED_EPISODE_DIR, "variation_descriptions.pkl"), "rb") as f:
        instructions = pickle.load(f)
    with open(os.path.join(FAILED_EPISODE_DIR, "low_dim_obs.pkl"), "rb") as f:
        data_demo = pickle.load(f)

    mock_obs_0 = data_demo[0].__dict__
    estimator = ProgressEstimator(
        manipulate_obj=config["semantic_mo_name"], 
        target_obj=config["semantic_to_name"]
    )


    # State tracking
    prev_obs_dict = mock_obs_0 
    prev_graph = extract_momagraph_frame(
            task_name=task_name,
            obs_dict=mock_obs_0, 
            init_obs_dict=mock_obs_0, 
            indices=config["indices"], 
            extents=extents_dict,
            instructions=instructions,
            prev_obs_dict=prev_obs_dict
        )
    prev_printed_progress = 0.0

    for t, obs in enumerate(data_demo):
        curr_graph = extract_momagraph_frame(
            task_name=task_name,
            obs_dict=obs.__dict__, 
            init_obs_dict=mock_obs_0, 
            indices=config["indices"], 
            extents=extents_dict,
            instructions=instructions,
            prev_obs_dict=prev_obs_dict
        )
        progress = estimator.estimate(curr_graph)

        if t == 0:
            print(f"[Frame {t:03d}] progress: {progress*100:>5.1f}% | Initialized.")
            print(json.dumps(curr_graph, indent=2, ensure_ascii=False))
            continue
        
        diffs = get_graph_diff(prev_graph, curr_graph, config)
        
        # Log if graph changes or progress increases significantly
        if diffs or (progress - prev_printed_progress >= 0.05) or progress == 1.0:
            print(f"\n[Frame {t:03d}] progress: {progress*100:>5.1f}%")
            if diffs:
                for d in diffs: print(f"  -> Relation Change: {d}")
                
            prev_printed_progress = progress
                
        prev_graph = curr_graph
        prev_obs_dict = obs.__dict__

    # Rendering
    render_trajectory_video(
        task_name=task_name, 
        episode_dir=FAILED_EPISODE_DIR, 
        config=config,
        output_name=f"failed_{task_name}_ep0.mp4"
    ) 
    print("\n[Complete] All analysis tasks finished.")