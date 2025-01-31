import os
import cv2
import json
import torch
import torchvision.transforms as T
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, predict
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node

"""
Hyper parameters
"""
# TEXT_PROMPT = "drivable ground. Rubber Corn. fluorescent lamp."
TEXT_PROMPT = "drivable ground."

# Dynamically find paths based on the script's location
# 스크립트 기준 베이스 경로
BASE_DIR = Path(__file__).resolve().parents[3] / "Grounded-SAM-2"
# 경로 설정 (PosixPath -> 문자열 변환)
# SAM2_CHECKPOINT = str(BASE_DIR / "checkpoints/sam2.1_hiera_large.pt")
# SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

SAM2_CHECKPOINT = str(BASE_DIR / "checkpoints/sam2.1_hiera_tiny.pt")
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

GROUNDING_DINO_CONFIG = str(BASE_DIR / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = str(BASE_DIR / "gdino_checkpoints/groundingdino_swint_ogc.pth")

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/isaac_sim_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the CV bridge
bridge = CvBridge()

class GroundedSAM2Node(Node):
    def __init__(self):
        super().__init__('grounded_sam2_node')
        self.subscription = self.create_subscription(
            Image,
            '/sim_camera/rgb',
            self.image_callback,
            10
        )
        self.sam2_predictor = self.build_sam2_predictor()
        self.grounding_model = self.build_grounding_model()

    def build_sam2_predictor(self):
        sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        return SAM2ImagePredictor(sam2_model)

    def build_grounding_model(self):
        return load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=DEVICE
        )

    def image_callback(self, msg):
        # Convert ROS2 Image to OpenCV format
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # numpy.ndarray
        h, w, _ = cv_image.shape

        # Convert numpy image to PyTorch tensor
        transform = T.Compose([
            T.ToTensor(),  # Convert numpy image to tensor
        ])
        image_tensor = transform(cv_image).unsqueeze(0).to(DEVICE)  # Add batch dimension and move to DEVICE

        # Set the image in the SAM2 predictor
        self.sam2_predictor.set_image(cv_image)

        # Grounding DINO prediction
        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image_tensor[0],  # Pass the PyTorch tensor
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        # Scale boxes to image dimensions
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Get masks from SAM2
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Convert masks shape for visualization
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Generate detections for visualization
        class_ids = np.array(list(range(len(labels))))
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )

        # Annotate the image
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=cv_image.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        labels = [f"{label} {confidence:.2f}" for label, confidence in zip(labels, confidences.numpy())]
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        # Show the annotated image
        cv2.imshow('Grounded SAM 2 - Drivable Ground', annotated_frame)
        cv2.waitKey(1)

        # Optionally save the result
        cv2.imwrite(str(OUTPUT_DIR / "last_annotated_frame.jpg"), annotated_frame)


def main(args=None):
    rclpy.init(args=args)
    node = GroundedSAM2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
