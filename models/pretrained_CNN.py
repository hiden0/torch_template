import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    mobilenet_v2,
    MobileNet_V2_Weights,
    vgg16,
    VGG16_Weights,
    resnet50,
    ResNet50_Weights,
    densenet121,
    DenseNet121_Weights,
    inception_v3,
    Inception_V3_Weights,
)


class ResNet50(nn.Module):
    def __init__(self, num_classes, train_layers=1, pretrained=True):
        super(ResNet50, self).__init__()
        self.backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )  # Usar ResNet-50 preentrenada si pretrained=True
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, num_classes
        )  # Reemplazar la capa final

        # Congelar los parámetros de las capas preentrenadas
        if train_layers == "all":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif train_layers == 1 or train_layers == 2 or train_layers == 3:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Descongelar las últimas capas según el parámetro train_layers
            if train_layers >= 1:
                for param in self.backbone.fc.parameters():
                    param.requires_grad = True
            if train_layers >= 2:
                for param in list(self.backbone.layer4.parameters()):
                    param.requires_grad = True
            if train_layers >= 3:
                for param in list(self.backbone.layer3.parameters()):
                    param.requires_grad = True

        else:
            raise ValueError("train_layers must be 'all', 1, 2 or 3")

    def forward(self, x):
        return self.backbone(x)


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes, train_layers=1, pretrained=True):
        super(EfficientNetB0, self).__init__()
        self.backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )  # EfficientNet-B0 preentrenada si pretrained=True
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, num_classes
        )  # Reemplazar la capa final

        if train_layers == "all":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif train_layers == 1 or train_layers == 2 or train_layers == 3:
            # Congelar los parámetros de las capas preentrenadas
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Descongelar las últimas capas según el parámetro train_layers
            if train_layers >= 1:
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True
            if train_layers >= 2:
                for param in list(self.backbone.features[-1].parameters()):
                    param.requires_grad = True
            if train_layers >= 3:
                for param in list(self.backbone.features[-2].parameters()):
                    param.requires_grad = True
        else:
            raise ValueError("train_layers must be 'all', 1, 2 or 3")

    def forward(self, x):
        return self.backbone(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, train_layers=1, pretrained=True):
        super(MobileNetV2, self).__init__()
        self.backbone = mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )  # MobileNetV2 preentrenada si pretrained=True
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, num_classes
        )  # Reemplazar la capa final

        if train_layers == "all":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif train_layers == 1 or train_layers == 2 or train_layers == 3:
            # Congelar los parámetros de las capas preentrenadas
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Descongelar las últimas capas según el parámetro train_layers
            if train_layers >= 1:
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True
            if train_layers >= 2:
                for param in list(self.backbone.features[-1].parameters()):
                    param.requires_grad = True
            if train_layers >= 3:
                for param in list(self.backbone.features[-2].parameters()):
                    param.requires_grad = True
        else:
            raise ValueError("train_layers must be 'all', 1, 2 or 3")

    def forward(self, x):
        return self.backbone(x)


class VGG16(nn.Module):
    def __init__(self, num_classes, train_layers=1, pretrained=True):
        super(VGG16, self).__init__()
        self.backbone = vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        )  # VGG-16 preentrenada si pretrained=True
        self.backbone.classifier[6] = nn.Linear(
            self.backbone.classifier[6].in_features, num_classes
        )  # Reemplazar la capa final

        if train_layers == "all":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif train_layers == 1 or train_layers == 2 or train_layers == 3:
            # Congelar los parámetros de las capas preentrenadas
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Descongelar las últimas capas según el parámetro train_layers
            if train_layers >= 1:
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True
            if train_layers >= 2:
                for param in list(self.backbone.features[-1].parameters()):
                    param.requires_grad = True
            if train_layers >= 3:
                for param in list(self.backbone.features[-2].parameters()):
                    param.requires_grad = True
        else:
            raise ValueError("train_layers must be 'all', 1, 2 or 3")

    def forward(self, x):
        return self.backbone(x)


class DenseNet121(nn.Module):
    def __init__(self, num_classes, train_layers=1, pretrained=True):
        super(DenseNet121, self).__init__()
        self.backbone = densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )  # DenseNet-121 preentrenada si pretrained=True
        self.backbone.classifier = nn.Linear(
            self.backbone.classifier.in_features, num_classes
        )  # Reemplazar la capa final

        if train_layers == "all":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif train_layers == 1 or train_layers == 2 or train_layers == 3:
            # Congelar los parámetros de las capas preentrenadas
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Descongelar las últimas capas según el parámetro train_layers
            if train_layers >= 1:
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True
            if train_layers >= 2:
                for param in list(self.backbone.features[-1].parameters()):
                    param.requires_grad = True
            if train_layers >= 3:
                for param in list(self.backbone.features[-2].parameters()):
                    param.requires_grad = True
        else:
            raise ValueError("train_layers must be 'all', 1, 2 or 3")

    def forward(self, x):
        return self.backbone(x)


class InceptionV3(nn.Module):
    def __init__(self, num_classes, train_layers=1, pretrained=True):
        super(InceptionV3, self).__init__()
        self.backbone = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
            aux_logits=True,  # Mantener aux_logits=True para evitar errores
        )
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, num_classes
        )  # Reemplazar la capa final
        self.backbone.AuxLogits.fc = nn.Linear(
            self.backbone.AuxLogits.fc.in_features, num_classes
        )  # Ajustar la capa auxiliar

        # Congelar los parámetros de las capas preentrenadas
        if train_layers == "all":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif train_layers in {1, 2, 3}:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if train_layers >= 1:
                for param in self.backbone.fc.parameters():
                    param.requires_grad = True
                for param in self.backbone.AuxLogits.fc.parameters():
                    param.requires_grad = True
            if train_layers >= 2:
                for param in self.backbone.Mixed_7c.parameters():
                    param.requires_grad = True
            if train_layers >= 3:
                for param in self.backbone.Mixed_7b.parameters():
                    param.requires_grad = True
        else:
            raise ValueError("train_layers must be 'all', 1, 2, or 3")

    def forward(self, x):
        x, aux = self.backbone(x)
        return x
