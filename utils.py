import torch
import torchvision
import cv2
import numpy as np

# Non-Maximum Suppression 

def non_max_suppression(predictions, conf_thresh=0.5, iou_thresh=0.4):
    """
    predictions: [N, 5+C] → (x, y, w, h, obj, class_probs)
    Returns: list of final boxes [x1, y1, x2, y2, score, class_id]
    """
    # Step 1: Filter by confidence
    scores = predictions[..., 4] * predictions[..., 5:].max(1)[0]  # obj * max class prob
    mask = scores > conf_thresh
    predictions = predictions[mask]
    scores = scores[mask]

    if predictions.size(0) == 0:
        return []

    # Step 2: Convert (x,y,w,h) to (x1,y1,x2,y2)
    boxes = predictions[:, :4].clone()
    boxes[:, 0] = predictions[:, 0] - predictions[:, 2] / 2  # x1
    boxes[:, 1] = predictions[:, 1] - predictions[:, 3] / 2  # y1
    boxes[:, 2] = predictions[:, 0] + predictions[:, 2] / 2  # x2
    boxes[:, 3] = predictions[:, 1] + predictions[:, 3] / 2  # y2

    class_ids = predictions[:, 5:].argmax(1)

    # Step 3: Perform NMS
    keep = torchvision.ops.nms(boxes, scores, iou_thresh)
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    # Final detections
    detections = []
    for i in range(len(boxes)):
        detections.append([
            boxes[i, 0].item(), boxes[i, 1].item(),
            boxes[i, 2].item(), boxes[i, 3].item(),
            scores[i].item(), int(class_ids[i].item())
        ])
    return detections



# Visualization using OpenCV

def draw_detections(image, detections, class_names):
    for (x1, y1, x2, y2, score, class_id) in detections:
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # helmet vs no-helmet
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_names[class_id]} {score:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


