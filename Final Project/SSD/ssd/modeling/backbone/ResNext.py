import torch

# inspired by: https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
from torchvision.models import resnext50_32x4d

# https://pytorch.org/hub/pytorch_vision_resnext/
class ResNext(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        #model = torch.hub.load('pytorch/vision:v0.9.0', 'resnext50_32x4d', pretrained=bool(cfg.MODEL.BACKBONE.PRETRAINED))
        model = resnext50_32x4d(pretrained=bool(cfg.MODEL.BACKBONE.PRETRAINED))

        # strip model of fc and pool layers at end
        self.model = torch.nn.Sequential(*(list(model.children())[:-2]))
        print("layers---------------------------")
        for i in range(len(self.model)):
            print("layer {}:\n {}".format(i, self.model[i]))


    def forward(self, x):


        out_features = []
        x = self.model[:6](x)
        out_features.append(x)
        x = self.model[6](x)
        out_features.append(x)
        x = self.model[7](x)
        out_features.append(x)


        return tuple(out_features)

