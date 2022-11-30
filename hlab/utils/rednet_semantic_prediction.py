import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint


mapping_mp3d_to_thda = {
    3: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    10: 6,
    11: 7,
    13: 8,
    14: 9,
    15: 10,
    18: 11,
    19: 12,
    20: 13,
    22: 14,
    23: 15,
    25: 16,
    26: 17,
    27: 18,
    33: 19,
    34: 20,
    38: 21,
    43: 22,  #  ('foodstuff', 42, task_cat: 21)
    44: 28,  #  ('stationery', 43, task_cat: 22)
    45: 26,  #  ('fruit', 44, task_cat: 23)
    46: 25,  #  ('plaything', 45, task_cat: 24)
    47: 24,  # ('hand_tool', 46, task_cat: 25)
    48: 23,  # ('game_equipment', 47, task_cat: 26)
    49: 27,  # ('kitchenware', 48, task_cat: 27)
}
category_to_task_category_id = {
    "chair": 0,
    "table": 1,
    "picture": 2,
    "cabinet": 3,
    "cushion": 4,
    "sofa": 5,
    "bed": 6,
    "chest_of_drawers": 7,
    "plant": 8,
    "sink": 9,
    "toilet": 10,
    "stool": 11,
    "towel": 12,
    "tv_monitor": 13,
    "shower": 14,
    "bathtub": 15,
    "counter": 16,
    "fireplace": 17,
    "gym_equipment": 18,
    "seating": 19,
    "clothes": 20,
}
THDA_BACKGROUND = 0
mapping_thda_to_objectnav = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
}


class RedNet(nn.Module):
    def __init__(self, cfg):

        super(RedNet, self).__init__()

        num_classes = cfg["n_classes"]
        pretrained = cfg["resnet_pretrained"]

        block = Bottleneck
        transblock = TransBasicBlock
        layers = [3, 4, 6, 3]
        # original resnet
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # resnet for depth channel
        self.inplanes = 64
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)

        self.agant0 = self._make_agant_layer(64, 64)
        self.agant1 = self._make_agant_layer(64 * 4, 64)
        self.agant2 = self._make_agant_layer(128 * 4, 128)
        self.agant3 = self._make_agant_layer(256 * 4, 256)
        self.agant4 = self._make_agant_layer(512 * 4, 512)

        # final block
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)

        self.final_deconv_custom = nn.ConvTranspose2d(
            self.inplanes, num_classes, kernel_size=2, stride=2, padding=0, bias=True
        )

        self.out5_conv_custom = nn.Conv2d(
            256, num_classes, kernel_size=1, stride=1, bias=True
        )
        self.out4_conv_custom = nn.Conv2d(
            128, num_classes, kernel_size=1, stride=1, bias=True
        )
        self.out3_conv_custom = nn.Conv2d(
            64, num_classes, kernel_size=1, stride=1, bias=True
        )
        self.out2_conv_custom = nn.Conv2d(
            64, num_classes, kernel_size=1, stride=1, bias=True
        )

    @classmethod
    def from_config(cls, cfg):
        model_cfg = {}
        model_cfg["n_classes"] = len(mapping_mp3d_to_thda) + 1
        model_cfg["resnet_pretrained"] = False
        return cls(model_cfg)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transpose(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes,
                    planes,
                    kernel_size=2,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def _make_agant_layer(self, inplanes, planes):

        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        return layers

    def forward_downsample(self, rgb, depth):
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu(depth)

        fuse0 = x + depth

        x = self.maxpool(fuse0)
        depth = self.maxpool(depth)

        # block 1
        x = self.layer1(x)
        depth = self.layer1_d(depth)
        fuse1 = x + depth
        # block 2
        x = self.layer2(fuse1)
        depth = self.layer2_d(depth)
        fuse2 = x + depth
        # block 3
        x = self.layer3(fuse2)
        depth = self.layer3_d(depth)
        fuse3 = x + depth
        # block 4
        x = self.layer4(fuse3)
        depth = self.layer4_d(depth)
        fuse4 = x + depth

        return fuse0, fuse1, fuse2, fuse3, fuse4

    def forward_upsample(self, fuse0, fuse1, fuse2, fuse3, fuse4):

        agant4 = self.agant4(fuse4)
        # upsample 1
        x = self.deconv1(agant4)
        if self.training:
            out5 = self.out5_conv_custom(x)
        x = x + self.agant3(fuse3)
        # upsample 2
        x = self.deconv2(x)
        if self.training:
            out4 = self.out4_conv_custom(x)
        x = x + self.agant2(fuse2)
        # upsample 3
        x = self.deconv3(x)
        if self.training:
            out3 = self.out3_conv_custom(x)
        x = x + self.agant1(fuse1)
        # upsample 4
        x = self.deconv4(x)
        if self.training:
            out2 = self.out2_conv_custom(x)
        x = x + self.agant0(fuse0)
        # final
        x = self.final_conv(x)
        out = self.final_deconv_custom(x)

        if self.training:
            return out, out2, out3, out4, out5

        return out

    def forward(self, rgb, depth, phase_checkpoint=False):

        if phase_checkpoint:
            depth.requires_grad_()
            fuses = checkpoint(self.forward_downsample, rgb, depth)
            out = checkpoint(self.forward_upsample, *fuses)
        else:
            fuses = self.forward_downsample(rgb, depth)
            out = self.forward_upsample(*fuses)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TransBasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=1,
                bias=False,
            )
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class SemanticPredRedNet:
    """
    Directly take a batch of RGB images (tensors) and get batched predictions.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # Setup model
        self.model = RedNet.from_config(cfg)
        self.model.eval()

        ckpt = torch.load(self.cfg.sem_pred_weights)
        model_state = {
            k.replace("module.", ""): v for k, v in ckpt["model_state"].items()
        }
        self.model.load_state_dict(model_state)
        # Convert model to device
        sem_gpu_id = self.cfg.sem_gpu_id
        if sem_gpu_id == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{sem_gpu_id}")
        self.model.to(self.device)
        # Normalization
        self.input_max_depth = self.cfg.max_depth
        self.input_min_depth = self.cfg.min_depth
        self.reqd_max_depth = 10.0
        self.reqd_min_depth = 0.0
        self.rgb_mean = torch.Tensor([0.493, 0.468, 0.438]).to(self.device)
        self.rgb_std = torch.Tensor([0.544, 0.521, 0.499]).to(self.device)
        self.depth_mean = torch.Tensor([0.213]).to(self.device)
        self.depth_std = torch.Tensor([0.285]).to(self.device)
        self.rgb_mean = rearrange(self.rgb_mean, "c -> () c () ()")
        self.rgb_std = rearrange(self.rgb_std, "c -> () c () ()")
        self.depth_mean = rearrange(self.depth_mean, "c -> () c () ()")
        self.depth_std = rearrange(self.depth_std, "c -> () c () ()")

    def get_predictions(self, batched_rgb, batched_depth):
        """
        Inputs:
            batched_rgb - (B, 3, H, W) RGB float Tensor values in [0.0, 255.0]
            batched_depth - (B, 1, H, W) depth float Tensor values in meters
        Outputs: (B, N, H, W) segmentation masks
        """
        _, _, H, W = batched_rgb.shape
        raw_depth = batched_depth
        with torch.no_grad():
            batched_depth = self.normalize_depth(batched_depth)
            batched_rgb = self.normalize_rgb(batched_rgb)
            predictions = self.model(batched_rgb, batched_depth)
            # Convert predictions to probabilities
            predictions = F.softmax(predictions, dim=1)
        semantic_inputs = self.process_predictions(predictions, raw_depth)
        return semantic_inputs

    def normalize_depth(self, depth):
        ########################################################################
        # Fix range of depth values
        ########################################################################
        # Convert depth to the required 0.0 -> 1.0 range
        mind, maxd = self.reqd_min_depth, self.reqd_max_depth
        depth = torch.clamp((depth - mind) / (maxd - mind), 0.0, 1.0)
        # Perform normalization
        depth.sub_(self.depth_mean).div_(self.depth_std)
        return depth

    def normalize_rgb(self, rgb):
        rgb = rgb / 255.0
        rgb.sub_(self.rgb_mean).div_(self.rgb_std)
        return rgb

    def process_predictions(self, predictions, raw_depth):
        B, N, H, W = predictions.shape
        semantic_inputs = torch.zeros(
            B, self.cfg.n_classes + 1, H, W, device=predictions.device
        )
        is_confident = torch.any(
            predictions >= self.cfg.sem_pred_prob_thr, dim=1
        )  # (B, H, W)
        predictions_argmax = torch.argmax(predictions, dim=1)  # (B, H, W)
        # Ignore predictions that are lower than the threshold
        predictions_argmax[~is_confident] = THDA_BACKGROUND  # Set to background class
        # Ignore predictions for pixels that are outside the depth threshold
        is_within_thresh = (raw_depth[:, 0] >= self.cfg.depth_thresh[0]) & (
            raw_depth[:, 0] <= self.cfg.depth_thresh[1]
        )
        predictions_argmax[~is_within_thresh] = THDA_BACKGROUND
        for i in range(len(mapping_mp3d_to_thda) + 1):
            if i not in mapping_thda_to_objectnav.keys():
                continue
            class_idx = mapping_thda_to_objectnav[i]
            semantic_inputs[:, class_idx] = predictions_argmax == i
        # Set background category
        semantic_inputs[:, self.cfg.n_classes] = predictions_argmax == THDA_BACKGROUND
        return semantic_inputs
