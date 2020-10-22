"""
Models

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torchvision.ops.boxes as box_ops

from torch import nn
from torchvision.ops._utils import _cat
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import transform

import pocket.models as models
from pocket.ops import Flatten, generate_masks

def LIS(x, T=8.3, k=12, w=10):
    """
    Low-grade suppression
    https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network
    """
    return T / ( 1 + torch.exp(k - w * x)) 

class InteractGraph(nn.Module):
    def __init__(self,
                out_channels,
                roi_pool_size,
                node_encoding_size, 
                representation_size, 
                num_cls, human_idx,
                object_class_to_target_class,
                box_feature_size=1024,
                fg_iou_thresh=0.5,
                num_iter=1):

        super().__init__()

        self.out_channels = out_channels
        self.roi_pool_size = roi_pool_size
        self.node_encoding_size = node_encoding_size
        self.representation_size = representation_size

        self.num_cls = num_cls
        self.human_idx = human_idx
        self.object_class_to_target_class = object_class_to_target_class

        self.fg_iou_thresh = fg_iou_thresh
        self.num_iter = num_iter

        # Box head to map RoI features to low dimensional
        self.box_head = nn.Sequential(
            Flatten(start_dim=1),
            nn.Linear(out_channels * roi_pool_size ** 2, box_feature_size),
            nn.ReLU(),
            nn.Linear(box_feature_size, box_feature_size),
            nn.ReLU()
        )

        # Spatial head to process spatial encodings
        self.box_spatial_head = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, 5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Second block
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Flatten(start_dim=1)    # Nx1024
        )

        self.spatial_attention = nn.Linear(1024, 1024, bias=False)

        # Compute adjacency matrix
        self.adjacency = nn.Sequential(
            nn.Linear((node_encoding_size + 1024)*2, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        # Compute messages
        self.hum_msg = nn.Sequential(
            nn.Linear(node_encoding_size, representation_size),
            nn.ReLU()
        )
        self.obj_msg = nn.Sequential(
            nn.Linear(node_encoding_size, representation_size),
            nn.ReLU()
        )

        # Update node hidden states
        self.hum_update = nn.Linear(
            node_encoding_size + representation_size,
            node_encoding_size,
            bias=False
        )
        self.obj_update = nn.Linear(
            node_encoding_size + representation_size,
            node_encoding_size,
            bias=False
        )

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets):
        """
        Arguements:
            boxes_h(Tensor[N, 4])
            boxes_o(Tensor[N, 4])
            targets(dict[Tensor]): Targets in an image with the following keys
                "boxes_h": Tensor[N, 4]
                "boxes_o": Tensor[N, 4)
                "labels": Tensor[N]
        """
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_cls, device=boxes_h.device)

        x, y = torch.nonzero(torch.min(
            box_ops.box_iou(boxes_h, targets["boxes_h"]),
            box_ops.box_iou(boxes_o, targets["boxes_o"])
        ) >= self.fg_iou_thresh).unbind(1)

        labels[x, targets["labels"][y]] = 1

        return labels

    def compute_prior_scores(self, x, y, scores, object_class):
        """
        Arguments:
            x(Tensor[M]): Indices of human boxes (paired)
            y(Tensor[M]): Indices of object boxes (paired)
            scores(Tensor[N])
            object_class(Tensor[N])
        """
        prior = torch.zeros(len(x), self.num_cls, device=scores.device)

        # Product of human and object detection scores with LIS
        prod = LIS(scores[x]) * LIS(scores[y])

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior[pair_idx, flat_target_idx] = prod[pair_idx]

        return prior

    @staticmethod
    def get_spatial_encoding(boxes, image_shapes, size=64):
        """
        Compute the spatial encoding of a bounding box by scaling the 
        longer side of the image to designated size and fill the area
        that a box covers with ones

        Arguments:
            box_coords(List[Tensor])
            image_shapes(List[Tuple[height, width]])
            size(int): Spatial resolution of the encoding
        """
        scaled_boxes = []
        # Rescale the boxes
        for boxes_per_image, shapes in zip(boxes, image_shapes):
            ratio = size / max(shapes)
            # Clamp the boxes to avoid coords. out of the image due to numerical error
            scaled_boxes.append(
                torch.clamp(boxes_per_image * ratio, 0, size)
            )

        scaled_boxes = torch.cat(scaled_boxes)
        device = scaled_boxes.device
        encodings = generate_masks(scaled_boxes.cpu(), size, size)

        return encodings.to(device)

    def forward(self, features, image_shapes, box_features, box_coords, box_labels, box_scores, targets=None):
        """
        Arguments:
            features(OrderedDict[Tensor]): Image pyramid with different levels
            box_features(Tensor[M, R])
            image_shapes(List[Tuple[height, width]])
            box_coords(List[Tensor])
            box_labels(List[Tensor])
            box_scores(List[Tensor])
            targets(list[dict]): Interaction targets with the following keys
                "boxes_h": Tensor[N, 4]
                "boxes_o": Tensor[N, 4]
                "labels": Tensor[N]
        Returns:
            all_box_pair_features(list[Tensor])
            all_boxes_h(list[Tensor])
            all_boxes_o(list[Tensor])
            all_object_class(list[Tensor])
            all_labels(list[Tensor])
            all_prior(list[Tensor])
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"

        box_features = self.box_head(box_features)
        box_masks = self.get_spatial_encoding(box_coords, image_shapes)
        box_spatial = self.box_spatial_head(box_masks[:, None, :, :])

        num_boxes = [len(boxes_per_image) for boxes_per_image in box_coords]
        
        counter = 0
        all_boxes_h = []; all_boxes_o = []; all_object_class = []
        all_labels = []; all_prior = []
        all_box_pair_features = []
        for b_idx, (coords, labels, scores) in enumerate(zip(box_coords, box_labels, box_scores)):
            n = num_boxes[b_idx]
            device = box_features.device

            n_h = torch.sum(labels == self.human_idx).item()
            # Skip image when there are no detected human or object instances
            # and when there is only one detected instance
            if n_h == 0 or n <= 1:
                continue
            if not torch.all(labels[:n_h]==self.human_idx):
                raise AssertionError("Human detections are not permuted to the top")

            node_encodings = box_features[counter: counter+n]
            node_spatial = box_spatial[counter: counter+n]
            # Duplicate human nodes
            h_node_encodings = node_encodings[:n_h]
            h_node_spatial = node_spatial[:n_h]
            # Get the pairwise index between every human and object instance
            x, y = torch.meshgrid(
                torch.arange(n_h, device=device),
                torch.arange(n, device=device)
            )
            # Remove pairs consisting of the same human instance
            x_keep, y_keep = torch.nonzero(x != y).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                continue
            # Human nodes have been duplicated and will be treated independently
            # of the humans included amongst object nodes
            x = x.flatten(); y = y.flatten()
            # Repeat spatial encodings [n_h, n, 1024]
            node_spatial_repeat = node_spatial.repeat(n_h, 1, 1)
            h_node_spatial_repeat = h_node_spatial.repeat(n, 1, 1).permute([1, 0, 2])
            # Compute directional encodings
            h_to_o_spatial = self.spatial_attention(
                (node_spatial_repeat - h_node_spatial_repeat).reshape(-1, 1024)
            ).reshape(n_h, n, 1024)
            o_to_h_spatial = -h_to_o_spatial

            adjacency_matrix = torch.ones(n_h, n, device=device)
            for i in range(self.num_iter):
                # Compute weights of each edge
                weights = self.adjacency(torch.cat([
                    torch.cat([h_node_encodings[x], h_node_spatial[x]], 1),
                    torch.cat([node_encodings[y]], node_spatial[y], 1)
                ], 1))
                adjacency_matrix = weights.reshape(n_h, n)
                # Compute object node to human node messages
                obj_message = self.obj_msg(node_encodings).repeat(
                    n_h, 1, 1) * o_to_h_spatial
                # Update human nodes
                h_node_encodings = self.hum_update(torch.cat([
                    h_node_encodings,
                    (adjacency_matrix[..., None] * obj_message).sum(1)
                ], 1))
                # Compute human node to object node messages
                hum_message = self.hum_msg(h_node_encodings).repeat(
                    n, 1, 1).permute([1, 0, 2]) * h_to_o_spatial
                # Update object nodes (including human nodes)
                node_encodings = self.obj_update(torch.cat([
                    node_encodings,
                    (adjacency_matrix[..., None] * hum_message).sum(0)
                ], 1))

            if targets is not None:
                all_labels.append(self.associate_with_ground_truth(
                    coords[x_keep], coords[y_keep], targets[b_idx])
                )
                
            all_box_pair_features.append(torch.cat([
                torch.cat([h_node_encodings[x_keep], h_node_spatial[x_keep]], 1),
                torch.cat([node_encodings[y_keep], node_spatial[y_keep]], 1)
            ], 1))
            all_boxes_h.append(coords[x_keep])
            all_boxes_o.append(coords[y_keep])
            all_object_class.append(labels[y_keep])
            # The prior score is the product between edge weights and the
            # pre-computed object detection scores with LIS
            all_prior.append(
                adjacency_matrix[x_keep, y_keep, None] *
                self.compute_prior_scores(x_keep, y_keep, scores, labels)
            )

            counter += n

        return all_box_pair_features, all_boxes_h, all_boxes_o, all_object_class, all_labels, all_prior

class BoxPairPredictor(nn.Module):
    def __init__(self, input_size, representation_size, num_classes):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(input_size, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, num_classes)
        )
    def forward(self, x, prior):
        return torch.sigmoid(self.predictor(x)) * prior

class InteractGraphNet(models.GenericHOINetwork):
    def __init__(self,
            object_to_action, human_idx,
            # Backbone parameters
            backbone_name="resnet50", pretrained=True,
            # Pooler parameters
            output_size=7, sampling_ratio=2,
            # Box pair head parameters
            node_encoding_size=1024,
            representation_size=1024,
            num_classes=117,
            fg_iou_thresh=0.5,
            num_iterations=1,
            # Transformation parameters
            min_size=800, max_size=1333,
            image_mean=None, image_std=None,
            # Preprocessing parameters
            box_nms_thresh=0.5,
            max_human=10,
            max_object=10
            ):

        backbone = models.fasterrcnn_resnet_fpn(backbone_name,
            pretrained=pretrained).backbone

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=output_size,
            sampling_ratio=sampling_ratio
        )

        box_pair_head = InteractGraph(
            out_channels=backbone.out_channels,
            roi_pool_size=output_size,
            node_encoding_size=node_encoding_size,
            representation_size=representation_size,
            num_cls=num_classes,
            human_idx=human_idx,
            object_class_to_target_class=object_to_action,
            fg_iou_thresh=fg_iou_thresh,
            num_iter=num_iterations
        )

        box_pair_predictor = BoxPairPredictor(
            input_size=node_encoding_size * 2,
            representation_size=representation_size,
            num_classes=num_classes
        )

        interaction_head = models.InteractionHead(
            box_roi_pool=box_roi_pool,
            box_pair_head=box_pair_head,
            box_pair_predictor=box_pair_predictor,
            num_classes=num_classes,
            human_idx=human_idx,
            box_nms_thresh=box_nms_thresh,
            max_human=max_human,
            max_object=max_object
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = models.HOINetworkTransform(min_size, max_size,
            image_mean, image_std)

        super().__init__(backbone, interaction_head, transform)

    def state_dict(self):
        """Override method to only return state dict of the interaction head"""
        return self.interaction_head.state_dict()
    def load_state_dict(self, x):
        """Override method to only load state dict of the interaction head"""
        self.interaction_head.load_state_dict(x)
