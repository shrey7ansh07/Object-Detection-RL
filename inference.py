"""
DDQN Object Detection Inference Script
--------------------------------------
Loads a trained Double DQN (DDQN) model trained for object localization 
(e.g. gallbladder in ultrasound). Produces a step-by-step detection video 
and final visualization with IoU if ground truth is provided.
"""

import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T

# Import components
from ddqnagent import DoubleDQNAgent
from enviornment import DQNObjectDetectionEnv
from visual_feature_extractor import VisualFeatureExtractor

# ============================================================
# CONFIG
# ============================================================

VISUAL_EMBEDDING_DIM = 128
ACTION_NAMES = [
    "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN",
    "GROW_WIDTH", "SHRINK_WIDTH", "GROW_HEIGHT", "SHRINK_HEIGHT", "STOP"
]

# ============================================================
# UTILS
# ============================================================

def draw_box(image, box, color=(0, 255, 0), label=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    img = image.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def save_video(frames, out_path="outputs/ddqn_inference.mp4", fps=2):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"🎥 Video saved to: {out_path}")


def calculate_iou(box1, box2):
    """Compute IoU between two boxes."""
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    inter_w = max(0, x_max_inter - x_min_inter)
    inter_h = max(0, y_max_inter - y_min_inter)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


# ============================================================
# INFERENCE LOGIC
# ============================================================

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # --- Load feature extractor ---
    feature_extractor = VisualFeatureExtractor(output_dim=VISUAL_EMBEDDING_DIM).to(device)
    feature_extractor.eval()

    # --- Load trained DDQN agent ---
    # --- Load trained DDQN agent ---
    checkpoint = torch.load(args.checkpoint, map_location=device)
    print(f"✅ Loaded checkpoint: {args.checkpoint}")

    state_dim = 5 + VISUAL_EMBEDDING_DIM
    action_size = 9
    agent = DoubleDQNAgent(state_dim, action_size, device)

    def filter_state_dict(state_dict):
        """Remove feature_extractor-related weights from checkpoint."""
        return {k.replace("policy_net.", "").replace("target_net.", ""): v 
                for k, v in state_dict.items() if not k.startswith("feature_extractor.")}

    # Filter and load policy network weights
    policy_state_dict = filter_state_dict(checkpoint["policy_net_state_dict"])
    missing_p, unexpected_p = agent.policy_net.load_state_dict(policy_state_dict, strict=False)
    print(f"✅ Loaded policy_net weights ({len(policy_state_dict)} params)")
    if missing_p: print(f"⚠️ Missing keys (policy): {missing_p}")
    if unexpected_p: print(f"⚠️ Unexpected keys (policy): {unexpected_p}")

    # Filter and load target network weights
    target_state_dict = filter_state_dict(checkpoint["target_net_state_dict"])
    missing_t, unexpected_t = agent.target_net.load_state_dict(target_state_dict, strict=False)
    print(f"✅ Loaded target_net weights ({len(target_state_dict)} params)")
    if missing_t: print(f"⚠️ Missing keys (target): {missing_t}")
    if unexpected_t: print(f"⚠️ Unexpected keys (target): {unexpected_t}")

    # Set to eval mode
    agent.policy_net.eval()
    agent.target_net.eval()

    print(f"✅ DDQN model initialized and ready for inference")


    print(f"✅ Loaded DDQN weights (epoch {checkpoint.get('epoch', '?')})")

    # --- Load image ---
    image = np.array(Image.open(args.image).convert("RGB"))
    gt_box = None

    if args.ground_truth:
        try:
            gt_box = np.array([float(x) for x in args.ground_truth.split(",")])
            assert len(gt_box) == 4
        except:
            print("⚠️  Invalid ground truth format. Expected x1,y1,x2,y2")

    # --- Create environment ---
    env = DQNObjectDetectionEnv(
        image=image,
        ground_truth_box=gt_box if gt_box is not None else [0, 0, 1, 1],
        feature_extractor=feature_extractor,
        max_steps=args.max_steps,
        move_percentage=0.1,
        initial_box=args.initial_box,
        device=device
    )

    # --- Inference loop ---
    state = env.reset()
    done = False
    frames = []
    step_idx = 0

    base_img = cv2.cvtColor(env.image.copy(), cv2.COLOR_RGB2BGR)

    print("\n🔍 Running DDQN inference...")
    print("-" * 70)

    while not done:
        step_idx += 1
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = agent.policy_net(state_tensor)
        action = q_values.argmax(1).item()

        next_state, reward, done, info = env.step(action)

        # Visualization frame
        frame = base_img.copy()
        x1, y1, x2, y2 = env.current_box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if gt_box is not None:
            gx1, gy1, gx2, gy2 = gt_box.astype(int)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
        cv2.putText(frame, f"Step {step_idx}: {ACTION_NAMES[action]}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        frames.append(frame)

        # print(f"Step {step_idx:2d} | Action: {ACTION_NAMES[action]:12s} | "
        #       f"IoU: {info['iou']:.3f} | Reward: {reward:+.2f}")

        print(f"Step {step_idx:2d} | Action: {ACTION_NAMES[action]:12s} | IoU: {info['iou']:.3f} | Max Q: {q_values.max():+.3f}")

        state = next_state

    print("-" * 70)
    print(f"✅ Detection finished in {step_idx} steps")
    print(f"Final box: {env.current_box}")

    if gt_box is not None:
        iou = calculate_iou(env.current_box, gt_box)
        print(f"IoU with GT: {iou:.4f}")

    # --- Save visualization ---
    if not args.no_video:
        save_video(frames, out_path=args.video_path, fps=2)

    # --- Final visualization ---
    visualize_final_result(image, env.current_box, gt_box, args.save_viz)


def visualize_final_result(image, detected_box, gt_box=None, save_path=None):
    """Visualize detected vs. ground truth boxes."""
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)
    ax.axis("off")

    x1, y1, x2, y2 = detected_box
    ax.add_patch(
        patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='lime', facecolor='none', label='Detected')
    )

    if gt_box is not None:
        gx1, gy1, gx2, gy2 = gt_box
        ax.add_patch(
            patches.Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1, linewidth=3, edgecolor='red', facecolor='none', label='Ground Truth')
        )
        iou = calculate_iou(detected_box, gt_box)
        ax.text(10, 30, f"IoU: {iou:.3f}", color='white', fontsize=12,
                bbox=dict(facecolor='black', alpha=0.7))

    ax.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"🖼️ Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDQN Object Detection Inference")

    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints_ddqn/ddqn_checkpoint_epoch_12.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--image", type=str,
                        default="dataset/images/train/im00008.jpg",
                        help="Path to input image")
    parser.add_argument("--max_steps", type=int, default=20,
                        help="Maximum number of agent steps")
    parser.add_argument("--initial_box", type=str, default="center",
                        choices=["center", "random"],
                        help="Initial bounding box placement")
    parser.add_argument("--video_path", type=str,
                        default="outputs/ddqn_inference.mp4",
                        help="Output video path")
    parser.add_argument("--no_video", action="store_true",
                        help="Disable saving video")
    parser.add_argument("--save_viz", type=str,
                        default="outputs/ddqn_result_viz.jpg",
                        help="Path to save final visualization")
    parser.add_argument("--ground_truth", type=str, default="458.9103690685413,138.2811950790861,686.5659050966608,326.3444639718805",
                        help="Optional ground truth box as 'x1,y1,x2,y2'")

    args = parser.parse_args()

    run_inference(args)
