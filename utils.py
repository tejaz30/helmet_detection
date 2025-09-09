import torch
import cv2
import numpy as np

# -------------------------------
# 🔹 Non-Maximum Suppression (NMS)
# -------------------------------
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


# -------------------------------
# 🔹 Visualization with OpenCV
# -------------------------------
def draw_detections(image, detections, class_names):
    for (x1, y1, x2, y2, score, class_id) in detections:
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # helmet vs no-helmet
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_names[class_id]} {score:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


# -------------------------------
# 🔹 Example Inference Script
# -------------------------------
if __name__ == "__main__":
    from yolov3 import YOLOv3, decode_predictions  # make sure your model file is imported
    import torchvision

    # 1. Load model
    num_classes = 3  # helmet, no-helmet, person
    model = YOLOv3(num_classes=num_classes)
    model.eval()

    # 2. Load image
    img_path = "test.jpg"
    image = cv2.imread(img_path)
    input_dim = 416
    img_resized = cv2.resize(image, (input_dim, input_dim))
    img_tensor = torch.from_numpy(img_resized[..., ::-1].transpose(2, 0, 1)).float()
    img_tensor = img_tensor.unsqueeze(0) / 255.0  # [1,3,416,416]

    # 3. Forward pass
    out1, out2, out3 = model(img_tensor)

    # Anchors for YOLOv3 (predefined, 9 total → 3 per scale)
    anchors = [
        [(116, 90), (156, 198), (373, 326)],  # scale 1 (13x13)
        [(30, 61), (62, 45), (59, 119)],      # scale 2 (26x26)
        [(10, 13), (16, 30), (33, 23)]        # scale 3 (52x52)
    ]

    # 4. Decode predictions
    boxes1 = decode_predictions(out1, anchors[0], num_classes, input_dim)
    boxes2 = decode_predictions(out2, anchors[1], num_classes, input_dim)
    boxes3 = decode_predictions(out3, anchors[2], num_classes, input_dim)

    all_boxes = torch.cat([boxes1, boxes2, boxes3], dim=1).squeeze(0)  # [N, 5+C]

    # 5. Apply NMS
    detections = non_max_suppression(all_boxes, conf_thresh=0.5, iou_thresh=0.4)

    # 6. Draw detections
    class_names = ["helmet", "no-helmet", "person"]
    image_with_boxes = draw_detections(image.copy(), detections, class_names)

    # 7. Save or display result
    cv2.imshow("Detections", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output.jpg", image_with_boxes)
