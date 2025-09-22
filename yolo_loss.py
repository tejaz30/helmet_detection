import torch
import torch.nn as nn

class YOLOHelmetLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        """
        Loss for 2-class helmet detection.
        lambda_coord: weight for box regression
        lambda_noobj: weight for no-object confidence
        """
        super(YOLOHelmetLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")   # For box regression
        self.bce = nn.BCELoss(reduction="sum")   # For objectness
        self.ce  = nn.CrossEntropyLoss()         # For 2-class classification
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        """
        predictions: [B, N, 5+2]  -> (x, y, w, h, obj, 2-class probs)
        targets:     [B, N, 5+2]  -> same format, class one-hot
        """

        # Split predictions
        pred_boxes = predictions[..., 0:4]        # (x, y, w, h)
        pred_obj   = predictions[..., 4]          # objectness
        pred_cls   = predictions[..., 5:]         # class probabilities

        # Split targets
        true_boxes = targets[..., 0:4]
        true_obj   = targets[..., 4]
        true_cls   = targets[..., 5:].argmax(-1)  # 0=helmet, 1=no-helmet

  
        # Localization Loss
        box_loss = self.mse(pred_boxes, true_boxes)

        # Objectness Loss
      
        obj_loss = self.bce(pred_obj, true_obj)

       
        # Classification Loss
       
        # Only compute CE where an object exists
        if true_obj.sum() > 0:
            cls_loss = self.ce(pred_cls[true_obj == 1], true_cls[true_obj == 1])
        else:
            cls_loss = torch.tensor(0.0, device=predictions.device)

      
        # Final weighted loss
        
        total_loss = (
            self.lambda_coord * box_loss +
            obj_loss +
            cls_loss
        )

        return total_loss, {
            "box_loss": box_loss.item(),
            "obj_loss": obj_loss.item(),
            "cls_loss": cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss
        }
