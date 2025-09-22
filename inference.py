import torch
import cv2
import numpy as np
from PIL import Image
from model import YOLOv3
from utils import non_max_suppression, draw_detections
from model import decode_predictions


def run_inference(weights_path, img_path, class_names, img_size=416, conf_thresh=0.5, iou_thresh=0.4):
    # Load model
    model = YOLOv3(num_classes=len(class_names))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    orig = np.array(img)
    img_resized = img.resize((img_size, img_size))
    x = torch.tensor(np.array(img_resized)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        out = model(x)  # raw output from network (single scale in your setup)

        # Decode predictions
        decoded = decode_predictions(out, num_classes=len(class_names), input_dim=img_size)

        # Non-max suppression
        detections = non_max_suppression(decoded[0], conf_thresh, iou_thresh)

    # Draw boxes on original image
    drawn = draw_detections(orig.copy(), detections, class_names)

    # Save result
    save_path = "inference_result.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
    print(f"Saved result to {save_path}")


if __name__ == "__main__":
    class_names = ["helmet", "no-helmet", "person"]  # adjust to your dataset
    run_inference(
        weights_path="checkpoints/yolo_helmet.pth",  # trained weights
        img_path="sample.jpg",                       # test image
        class_names=class_names
    )
