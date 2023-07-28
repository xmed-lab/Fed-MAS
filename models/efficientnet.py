import torch
from torch import nn
from torchvision.models import EfficientNet
from torchvision.models.efficientnet import _efficientnet_conf


class efficientnet_b0_modified(EfficientNet):
    def __init__(self, num_classes=1000, KD=False, projection=False, **kwargs):

        super(efficientnet_b0_modified, self).__init__(dropout=0.2, **kwargs)
        self.projection = projection
        self.KD = KD
        self.drop_out = torch.nn.Dropout(p=0.2, inplace=True)
        self.fc = nn.Linear(1280, num_classes)

        if projection:
            self.p1 = nn.Linear(1280, 1280)
            self.p2 = nn.Linear(1280, 640)
            self.fc = nn.Linear(640, num_classes)

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x_f = x.view(x.size(0), -1)  # B x 64

        if self.projection:
            x_p = self.p1(x_f)
            x_p = self.relu(x_p)
            x_f = self.p2(x_p)
        else:
            x = self.drop_out(x_f)

        x = self.fc(x)  # B x num_classes

        if self.KD == True:
            return x_f, x
        else:
            return x


class efficientnet_b0_F(EfficientNet):
    def __init__(self, num_classes=1000, KD=False, projection=False,free_model_output_dim=None, **kwargs):

        super(efficientnet_b0_F, self).__init__(dropout=0.2, **kwargs)
        self.projection = projection
        self.KD = KD
        self.drop_out = torch.nn.Dropout(p=0.2)
        self.fc = nn.Linear(1280, num_classes)
        self.free_model_output_dim=free_model_output_dim
        if projection:
            #not used in the output but to project the latent space features to space comparable with the free lunch model
            #it can be help highly imbalance datasets (HyperKvasir) and restrict balanced datasets
            #infact if used without a pre-trained model it can distort the backbone feature heavily
            #https://arxiv.org/abs/2202.10054
            if self.free_model_output_dim:
                self.p_free = nn.Sequential(
                    nn.Linear(1280, 1280),
                    nn.BatchNorm1d(1280),
                    nn.ReLU(inplace=True),
                    nn.Linear(1280, free_model_output_dim),
                )
                self.fc_free = nn.Linear(free_model_output_dim, num_classes)

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x_f = x.view(x.size(0), -1)  # B x 64


        if self.projection:
            x_p_free = self.p_free(x_f)
            x_free = self.fc_free(x_p_free)

        x = self.drop_out(x_f)
        x = self.fc(x)  # B x num_classes

        if self.KD == True:
            return x_f, x, x_p_free, x_free
        else:
            return x


def efficientnet_b0_normal(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)

    model = efficientnet_b0_modified(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel, num_classes=class_num, **kwargs)

    return model


def efficientnet_b0_free(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    model = efficientnet_b0_F(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel, num_classes=class_num, **kwargs)

    return model
