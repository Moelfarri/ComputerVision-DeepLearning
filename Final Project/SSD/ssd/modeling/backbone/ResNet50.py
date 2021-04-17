import torch
from torch import nn
import torchvision


class ResNet50(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.model = nn.Sequential(*(list(self.resnet.children()))[:-2])
        
        
        self.conv1 = self.model[6]
        
        self.conv2 = self.model[7]
        
        self.conv3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(output_channels[1]),
            nn.Conv2d(
                in_channels=output_channels[1],
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(
                in_channels=512,
                out_channels=output_channels[2],
                kernel_size=3,
                stride=2,
                padding=1),
        )


        self.conv4 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(output_channels[2]),
            nn.Conv2d(
                in_channels=output_channels[2],
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(
                in_channels=512,
                out_channels=output_channels[3],
                kernel_size=3,
                stride=2,
                padding=1),
        )


        self.conv5 =  nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(output_channels[3]),
            nn.Conv2d(
                in_channels=output_channels[3],
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(
                in_channels=512,
                out_channels=output_channels[4],
                kernel_size=3,
                stride=2,
                padding=1),
        )

        self.conv6 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(output_channels[4]),
            nn.Conv2d(
                in_channels=output_channels[4],
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(
                in_channels=512,
                out_channels=output_channels[5],
                kernel_size=3,
                stride=1,
                padding=0)
        )
            
       

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        #ResNet 50 first layers: 
        x = self.model[0:3](x) #skip first maxpool
        x = self.model[4](x)
        x = self.model[5](x)

        #SSD LAYERS:
        out_features = []
        x = self.conv1(x)
        out_features.append(x)
        x = self.conv2(x)
        out_features.append(x)
        x = self.conv3(x)
        out_features.append(x)
        x = self.conv4(x)
        out_features.append(x)
        x = self.conv5(x)
        out_features.append(x)
        x = self.conv6(x)
        out_features.append(x)

        

        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            # expected_shape = (out_channel, h, w)  # TODO bug in starter code?
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

    
