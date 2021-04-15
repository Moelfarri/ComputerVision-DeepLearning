import torch
from torch import nn
import torchvision


class ResNet(torch.nn.Module):
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

        
        self.resnet = torchvision.models.resnet18(pretrained=True)
        
        #-2 tar siste 2 layers
        self.model = nn.Sequential(*(list(self.resnet.children())[:-2]))
        
        
        # 5x5
        self.conv4 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(output_channels[2]),
            nn.Conv2d(
                in_channels=output_channels[2],
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,
                out_channels=output_channels[3],
                kernel_size=3,
                stride=2,
                padding=1),
        )

        # 3x3
        self.conv5 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(output_channels[3]),
            nn.Conv2d(
                in_channels=output_channels[3],
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,
                out_channels=output_channels[4],
                kernel_size=3,
                stride=2,
                padding=1),
        )

        # 1x1
        self.conv6 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(output_channels[4]),
            nn.Conv2d(
                in_channels=output_channels[4],
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,
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
               
        #get to second layer of ResNet18  
        out_features = []
        x = self.model[:6](x)
        out_features.append(x)
        x = self.model[6](x)
        out_features.append(x)
        x = self.model[7](x)
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

    

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.append("../../config")
    from defaults import cfg

    from PIL import Image
    from torchvision import transforms

    def get_parser():
        parser = argparse.ArgumentParser(description='Model Training With PyTorch')
        parser.add_argument(
            "config_file",
            default="",
            metavar="FILE",
            help="path to config file",
            type=str,
        )
        parser.add_argument(
                "opts",
            help="Modify config options using the command-line",
            default=None,
            nargs=argparse.REMAINDER,
        )
        return parser

    # to get feature map sizes, and output channel sizes
    # python3 resNext.py ../../../configs/resnext.yaml
    # python ResNet.py ../../../configs/train_rdd2020_resnet.yaml
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    cnn = ResNet(cfg)

    #Resize bildet akkurat som du vil her og se hvordan det outputes
    transform = transforms.Compose([
     transforms.Resize((300, 300)),
     transforms.ToTensor(),
     ])

    x = Image.open("dog.jpg")
    x = transform(x)
    x = torch.unsqueeze(x, 0)
    print("original shape: {}".format(x.shape))
    for layer in cnn.model:
        x = layer(x)
        print("shape: {}".format(x.shape))