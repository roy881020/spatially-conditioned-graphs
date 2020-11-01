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
            nn.Linear(out_channels * roi_pool_size ** 2, node_encoding_size),
            nn.ReLU(),
            nn.Linear(node_encoding_size, node_encoding_size),
            nn.ReLU()
        )

        # Compute adjacency matrix
        self.adjacency = nn.Sequential(
            nn.Linear(node_encoding_size*2, representation_size),
            nn.ReLU(),
            nn.Linear(representation_size, int(representation_size/2)),
            nn.ReLU(),
            nn.Linear(int(representation_size/2), 1),
            nn.Sigmoid()
        )

        # Compute messages
        self.sub_to_obj = nn.Sequential(
            nn.Linear(node_encoding_size, representation_size),
            nn.ReLU()
        )
        self.obj_to_sub = nn.Sequential(
            nn.Linear(node_encoding_size, representation_size),
            nn.ReLU()
        )

        # Update node hidden states
        self.sub_update = nn.Linear(
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
            # Duplicate human nodes
            h_node_encodings = node_encodings[:n_h]
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

            adjacency_matrix = torch.ones(n_h, n, device=device)
            for i in range(self.num_iter):
                # Compute weights of each edge
                weights = self.adjacency(torch.cat([
                    h_node_encodings[x],
                    node_encodings[y]
                ], 1))
                adjacency_matrix = weights.reshape(n_h, n)

                # Update human nodes
                h_node_encodings = self.sub_update(torch.cat([
                    h_node_encodings,
                    torch.mm(adjacency_matrix, self.obj_to_sub(node_encodings))
                ], 1))

                # Update object nodes (including human nodes)
                node_encodings = self.obj_update(torch.cat([
                    node_encodings,
                    torch.mm(adjacency_matrix.t(), self.sub_to_obj(h_node_encodings))
                ], 1))

            if targets is not None:
                all_labels.append(self.associate_with_ground_truth(
                    coords[x_keep], coords[y_keep], targets[b_idx])
                )
                
            all_box_pair_features.append(torch.cat([
                h_node_encodings[x_keep], node_encodings[y_keep]
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
    def forward(self, x):
        return self.predictor(x)

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

class ModelWithGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_pair_predictor = BoxPairPredictor(
            2048 + 117,
            1024,
            117
        )
    def forward(self, x):
        """
        x(List[dict])
            'features': Nx2048
            'boxes_h': Nx4
            'boxes_o': Nx4
            'object_class': N
            'labels': Nx117
            'prior': Nx117
            'index': 1
        """
        box_pair_features = torch.cat([
            x_per_image['features'] for x_per_image in x
        ])
        box_pair_prior = torch.cat([
            x_per_image['prior'] for x_per_image in x
        ])
        box_pair_labels = torch.cat([
            x_per_image['labels'] for x_per_image in x
        ])

        if len(box_pair_features) == 0:
            return None

        logits = self.box_pair_predictor(torch.cat([
            box_pair_features, box_pair_labels
        ], 1))
        i, j = torch.nonzero(box_pair_prior).unbind(1)

        results = [
            torch.sigmoid(logits[i, j]) * box_pair_prior[i, j],
            j, box_pair_labels[i, j],
        ]
        if self.training:
            loss = nn.functional.binary_cross_entropy(
                results[0], results[2]
            )
            results.append(loss)

        return results

class ModelWithOnlyGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_pair_predictor = BoxPairPredictor(
            117,
            1024,
            117
        )
    def forward(self, x):
        box_pair_prior = torch.cat([
            x_per_image['prior'] for x_per_image in x
        ])
        box_pair_labels = torch.cat([
            x_per_image['labels'] for x_per_image in x
        ])

        if len(box_pair_prior) == 0:
            return None

        logits = self.box_pair_predictor(box_pair_labels)
        i, j = torch.nonzero(box_pair_prior).unbind(1)

        results = [
            torch.sigmoid(logits[i, j]) * box_pair_prior[i, j],
            j, box_pair_labels[i, j],
        ]
        if self.training:
            loss = nn.functional.binary_cross_entropy(
                results[0], results[2]
            )
            results.append(loss)

        return results

class ModelWithNone(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_pair_predictor = BoxPairPredictor(
            2048,
            1024,
            117
        )
    def forward(self, x):
        box_pair_features = torch.cat([
            x_per_image['features'] for x_per_image in x
        ])
        box_pair_prior = torch.cat([
            x_per_image['prior'] for x_per_image in x
        ])
        box_pair_labels = torch.cat([
            x_per_image['labels'] for x_per_image in x
        ])

        if len(box_pair_features) == 0:
            return None

        logits = self.box_pair_predictor(box_pair_features)
        i, j = torch.nonzero(box_pair_prior).unbind(1)

        results = [
            torch.sigmoid(logits[i, j]) * box_pair_prior[i, j],
            j, box_pair_labels[i, j],
        ]
        if self.training:
            loss = nn.functional.binary_cross_entropy(
                results[0], results[2]
            )
            results.append(loss)

        return results

class ModelWith2Masks(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_pair_predictor = BoxPairPredictor(
            2048 + 2048,
            2048,
            117
        )
        self.spatial_head = nn.Sequential(
            nn.Conv2d(2, 64, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 5),
            nn.MaxPool2d(2),
            Flatten(start_dim=1),
            nn.Linear(5408, 2048)
        )

    @staticmethod
    def get_spatial_encoding(boxes_1, boxes_2, size=64):
        """
        Arguments:
            x(Tensor[M]): Indices of human boxes (paired)
            y(Tensor[M]): Indices of object boxes (paired)
            boxes(Tensor[N, 4])
            size(int): Spatial resolution of the encoding
        """
        device = boxes_1.device
        boxes_1 = boxes_1.clone().cpu()
        boxes_2 = boxes_2.clone().cpu()

        # Find the top left and bottom right corners
        top_left = torch.min(boxes_1[:, :2], boxes_2[:, :2])
        bottom_right = torch.max(boxes_1[:, 2:], boxes_2[:, 2:])
        # Shift
        boxes_1 -= top_left.repeat(1, 2)
        boxes_2 -= top_left.repeat(1, 2)
        # Scale
        ratio = size / (bottom_right - top_left)
        boxes_1 *= ratio.repeat(1, 2)
        boxes_2 *= ratio.repeat(1, 2)
        # Round to integer
        boxes_1.round_(); boxes_2.round_()
        boxes_1 = boxes_1.long()
        boxes_2 = boxes_2.long()

        spatial_encoding = torch.zeros(len(boxes_1), 2, size, size)
        for i, (b1, b2) in enumerate(zip(boxes_1, boxes_2)):
            spatial_encoding[i, 0, b1[1]:b1[3], b1[0]:b1[2]] = 1
            spatial_encoding[i, 1, b2[1]:b2[3], b2[0]:b2[2]] = 1

        return spatial_encoding.to(device)

    def forward(self, x):
        boxes_h = [x_per_image['boxes_h'] for x_per_image in x]
        boxes_o = [x_per_image['boxes_o'] for x_per_image in x]
        masks = self.get_spatial_encoding(
            torch.cat(boxes_h), torch.cat(boxes_o)
        )
        box_pair_spatial = self.spatial_head(masks)

        box_pair_features = torch.cat([
            x_per_image['features'] for x_per_image in x
        ])
        box_pair_prior = torch.cat([
            x_per_image['prior'] for x_per_image in x
        ])
        box_pair_labels = torch.cat([
            x_per_image['labels'] for x_per_image in x
        ])

        logits = self.box_pair_predictor(torch.cat([
            box_pair_features, box_pair_spatial
        ], 1))
        i, j = torch.nonzero(box_pair_prior).unbind(1)

        results = [
            torch.sigmoid(logits[i, j]) * box_pair_prior[i, j],
            j, box_pair_labels[i, j],
        ]
        if self.training:
            loss = nn.functional.binary_cross_entropy(
                results[0], results[2]
            )
            results.append(loss)

        return results

class ModelWith1Mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_pair_predictor = BoxPairPredictor(
            2048 + 2048,
            2048,
            117
        )
        # Spatial head to process spatial encodings
        self.box_spatial_head = nn.Sequential(
            # First block
            nn.Conv2d(2, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            # Second block
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(2),
            Flatten(start_dim=1)    # Nx1024
        )

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

    def forward(self, x):
        boxes_h = [x_per_image['boxes_h'] for x_per_image in x]
        boxes_o = [x_per_image['boxes_o'] for x_per_image in x]
        image_sizes = [x_per_image['size'] for x_per_image in x]

        boxes = []; indices = []
        for b_h, b_o in zip(boxes_h, boxes_o):
            b, idx = torch.cat([b_h, b_o]).unique(return_inverse=True, dim=0)
            boxes.append(b)
            indices.append(idx)
        masks = self.get_spatial_encoding(boxes, image_sizes)
        avg_masks = masks.mean(0)
        masks = torch.stack([
            masks,
            avg_masks.repeat(masks.shape[0], 1, 1)
        ], dim=1)
        spatial_features = self.box_spatial_head(masks)

        num_boxes = [len(b) for b in boxes]
        h_spatial = []; o_spatial = []
        for idx, f in zip(indices, spatial_features.split(num_boxes)):
            all_spatial = f[idx]
            n = int(len(all_spatial) / 2)
            h_spatial.append(all_spatial[:n])
            o_spatial.append(all_spatial[n:])
        h_spatial = torch.cat(h_spatial)
        o_spatial = torch.cat(o_spatial)

        box_pair_features = torch.cat([
            x_per_image['features'] for x_per_image in x
        ])
        box_pair_prior = torch.cat([
            x_per_image['prior'] for x_per_image in x
        ])
        box_pair_labels = torch.cat([
            x_per_image['labels'] for x_per_image in x
        ])

        logits = self.box_pair_predictor(torch.cat([
            box_pair_features, h_spatial, o_spatial
        ], 1))
        i, j = torch.nonzero(box_pair_prior).unbind(1)

        results = [
            torch.sigmoid(logits[i, j]) * box_pair_prior[i, j],
            j, box_pair_labels[i, j],
        ]
        if self.training:
            loss = nn.functional.binary_cross_entropy(
                results[0], results[2]
            )
            results.append(loss)

        return results

class ModelWithVec(nn.Module):
    def __init__(self):
        super().__init__()
        self.appearance_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU()
        )

        self.spatial_predictor = nn.Linear(512, 117)
        self.joint_predictor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 117)
        )

    @staticmethod
    def get_handcrafted_encodings(boxes_1, boxes_2, shapes, eps=1e-10):
        features = []
        for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
            w, h = shape

            c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
            c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

            b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
            b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

            d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
            d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

            iou = torch.diag(box_ops.box_iou(b1, b2))

            # Construct spatial encoding
            f = torch.stack([
                # Relative position of box centre
                c1_x / w, c1_y / h, c2_x / w, c2_y / h,
                # Relative box width and height
                b1_w / w, b1_h / h, b2_w / w, b2_h / h,
                # Relative box area
                b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
                b2_w * b2_h / (b1_w * b1_h + eps),
                # Box aspect ratio
                b1_w / (b1_h + eps), b2_w / (b2_h + eps),
                # Intersection over union
                iou,
                # Relative distance and direction of the object w.r.t. the person
                (c2_x > c1_x).float() * d_x,
                (c2_x < c1_x).float() * d_x,
                (c2_y > c1_y).float() * d_y,
                (c2_y < c1_y).float() * d_y,
            ], 1)

            features.append(
                torch.cat([f, torch.log(f + eps)], 1)
            )
        return torch.cat(features)

    def forward(self, x):
        boxes_h = [x_per_image['boxes_h'] for x_per_image in x]
        boxes_o = [x_per_image['boxes_o'] for x_per_image in x]
        image_sizes = [x_per_image['size'] for x_per_image in x]
        box_pair_spatial = self.get_handcrafted_encodings(
            boxes_h, boxes_o, image_sizes
        )
        box_pair_features = torch.cat([
            x_per_image['features'] for x_per_image in x
        ])
        box_pair_prior = torch.cat([
            x_per_image['prior'] for x_per_image in x
        ])
        box_pair_labels = torch.cat([
            x_per_image['labels'] for x_per_image in x
        ])

        f_s = self.spatial_head(box_pair_spatial)
        logits_s = self.spatial_predictor(f_s)
        f_a = self.appearance_head(box_pair_features)
        logits_c = self.joint_predictor(torch.cat([
            f_a, f_s
        ], 1))

        i, j = torch.nonzero(box_pair_prior).unbind(1)

        results = [
            torch.sigmoid(logits_s[i, j]) * torch.sigmoid(logits_c[i, j]) * box_pair_prior[i, j],
            j, box_pair_labels[i, j],
        ]
        if self.training:
            loss = nn.functional.binary_cross_entropy(
                results[0], results[2]
            )
            results.append(loss)

        return results

class ModelWithVecAtten(nn.Module):
    def __init__(self):
        super().__init__()
        self.appearance_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
        )

        self.spatial_predictor = nn.Linear(1024, 117)
        self.spatial_attention = nn.Linear(1024, 1024)
        self.joint_predictor = nn.Linear(1024, 117)

    def forward(self, x):
        boxes_h = [x_per_image['boxes_h'] for x_per_image in x]
        boxes_o = [x_per_image['boxes_o'] for x_per_image in x]
        image_sizes = [x_per_image['size'] for x_per_image in x]
        box_pair_spatial = ModelWithVec.get_handcrafted_encodings(
            boxes_h, boxes_o, image_sizes
        )
        box_pair_features = torch.cat([
            x_per_image['features'] for x_per_image in x
        ])
        box_pair_prior = torch.cat([
            x_per_image['prior'] for x_per_image in x
        ])
        box_pair_labels = torch.cat([
            x_per_image['labels'] for x_per_image in x
        ])

        f_s = self.spatial_head(box_pair_spatial)
        logits_s = self.spatial_predictor(f_s)
        f_a = self.appearance_head(box_pair_features)
        logits_c = self.joint_predictor(
            f_a * self.spatial_attention(f_s)
        )

        i, j = torch.nonzero(box_pair_prior).unbind(1)

        results = [
            torch.sigmoid(logits_s[i, j]) * torch.sigmoid(logits_c[i, j]) * box_pair_prior[i, j],
            j, box_pair_labels[i, j],
        ]
        if self.training:
            loss = nn.functional.binary_cross_entropy(
                results[0], results[2]
            )
            results.append(loss)

        return results

class ModelWithVecSum(nn.Module):
    def __init__(self):
        super().__init__()
        self.appearance_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU()
        )

        self.spatial_predictor = nn.Linear(512, 117)
        self.spatial_attention = nn.Linear(512, 512)
        self.joint_predictor = nn.Linear(512, 117)

    def forward(self, x):
        boxes_h = [x_per_image['boxes_h'] for x_per_image in x]
        boxes_o = [x_per_image['boxes_o'] for x_per_image in x]
        image_sizes = [x_per_image['size'] for x_per_image in x]
        box_pair_spatial = ModelWithVec.get_handcrafted_encodings(
            boxes_h, boxes_o, image_sizes
        )
        box_pair_features = torch.cat([
            x_per_image['features'] for x_per_image in x
        ])
        box_pair_prior = torch.cat([
            x_per_image['prior'] for x_per_image in x
        ])
        box_pair_labels = torch.cat([
            x_per_image['labels'] for x_per_image in x
        ])

        f_s = self.spatial_head(box_pair_spatial)
        logits_s = self.spatial_predictor(f_s)
        f_a = self.appearance_head(box_pair_features)
        logits_c = self.joint_predictor(
            f_a + self.spatial_attention(f_s)
        )

        i, j = torch.nonzero(box_pair_prior).unbind(1)

        results = [
            torch.sigmoid(logits_s[i, j]) * torch.sigmoid(logits_c[i, j]) * box_pair_prior[i, j],
            j, box_pair_labels[i, j],
        ]
        if self.training:
            loss = nn.functional.binary_cross_entropy(
                results[0], results[2]
            )
            results.append(loss)

        return results

class SpatialPairwiseMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_head = nn.Sequential(
            nn.Conv2d(2, 64, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 5),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            Flatten(start_dim=1),
            nn.Linear(32, 512),
            nn.ReLU()
        )
        self.box_pair_predictor = nn.Linear(512, 117)
    def forward(self, x):
        boxes_h = [x_per_image['boxes_h'] for x_per_image in x]
        boxes_o = [x_per_image['boxes_o'] for x_per_image in x]
        masks = ModelWith2Masks.get_spatial_encoding(
            torch.cat(boxes_h), torch.cat(boxes_o)
        )
        box_pair_spatial = self.spatial_head(masks)

        box_pair_prior = torch.cat([
            x_per_image['prior'] for x_per_image in x
        ])
        box_pair_labels = torch.cat([
            x_per_image['labels'] for x_per_image in x
        ])

        logits = self.box_pair_predictor(box_pair_spatial)
        i, j = torch.nonzero(box_pair_prior).unbind(1)

        results = [
            torch.sigmoid(logits[i, j]) * box_pair_prior[i, j],
            j, box_pair_labels[i, j],
        ]
        if self.training:
            loss = nn.functional.binary_cross_entropy(
                results[0], results[2]
            )
            results.append(loss)

        return results

class SpatialIndividualMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_spatial_head = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(2),
            Flatten(start_dim=1)    # Nx1024
        )
        self.box_pair_predictor = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 117)
        )

    def forward(self, x):
        boxes_h = [x_per_image['boxes_h'] for x_per_image in x]
        boxes_o = [x_per_image['boxes_o'] for x_per_image in x]
        image_sizes = [x_per_image['size'] for x_per_image in x]

        boxes = []; indices = []
        for b_h, b_o in zip(boxes_h, boxes_o):
            b, idx = torch.cat([b_h, b_o]).unique(return_inverse=True, dim=0)
            boxes.append(b)
            indices.append(idx)
        masks = ModelWith1Mask.get_spatial_encoding(boxes, image_sizes)
        avg_masks = masks.mean(0)
        masks = torch.stack([
            masks,
            avg_masks.repeat(masks.shape[0], 1, 1)
        ], dim=1)
        spatial_features = self.box_spatial_head(masks)

        num_boxes = [len(b) for b in boxes]
        h_spatial = []; o_spatial = []
        for idx, f in zip(indices, spatial_features.split(num_boxes)):
            all_spatial = f[idx]
            n = int(len(all_spatial) / 2)
            h_spatial.append(all_spatial[:n])
            o_spatial.append(all_spatial[n:])
        h_spatial = torch.cat(h_spatial)
        o_spatial = torch.cat(o_spatial)

        box_pair_prior = torch.cat([
            x_per_image['prior'] for x_per_image in x
        ])
        box_pair_labels = torch.cat([
            x_per_image['labels'] for x_per_image in x
        ])

        logits = self.box_pair_predictor(torch.cat([
            h_spatial, o_spatial
        ], 1))
        i, j = torch.nonzero(box_pair_prior).unbind(1)

        results = [
            torch.sigmoid(logits[i, j]) * box_pair_prior[i, j],
            j, box_pair_labels[i, j],
        ]
        if self.training:
            loss = nn.functional.binary_cross_entropy(
                results[0], results[2]
            )
            results.append(loss)

        return results

class SpatialHandcraft(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_pair_predictor = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 117)
        )

    def forward(self, x):
        boxes_h = [x_per_image['boxes_h'] for x_per_image in x]
        boxes_o = [x_per_image['boxes_o'] for x_per_image in x]
        image_sizes = [x_per_image['size'] for x_per_image in x]
        box_pair_spatial = ModelWithVec.get_handcrafted_encodings(
            boxes_h, boxes_o, image_sizes
        )
        
        box_pair_prior = torch.cat([
            x_per_image['prior'] for x_per_image in x
        ])
        box_pair_labels = torch.cat([
            x_per_image['labels'] for x_per_image in x
        ])

        logits = self.box_pair_predictor(box_pair_spatial)
        i, j = torch.nonzero(box_pair_prior).unbind(1)

        results = [
            torch.sigmoid(logits[i, j]) * box_pair_prior[i, j],
            j, box_pair_labels[i, j],
        ]
        if self.training:
            loss = nn.functional.binary_cross_entropy(
                results[0], results[2]
            )
            results.append(loss)

        return results
