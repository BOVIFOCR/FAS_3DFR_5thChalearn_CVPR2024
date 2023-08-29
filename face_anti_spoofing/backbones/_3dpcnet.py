import sys
import torch
import torch.nn as nn

from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200



class _3DPCNet(nn.Module):
    def __init__(self, encoder_name='r18', face_embedd_size=256, num_output_points=2500, num_axis=3, fp16=False, **kwargs):
        super(_3DPCNet, self).__init__()
        self.encoder_name = encoder_name
        self.face_embedd_size = face_embedd_size
        self.num_output_points = num_output_points
        self.num_axis = num_axis
        self.fp16 = fp16

        if self.encoder_name == "r18":
            self.encoder = iresnet18(False, **kwargs)
        elif self.encoder_name == "r34":
            self.encoder = iresnet34(False, **kwargs)
        elif self.encoder_name == "r50":
            self.encoder = iresnet50(False, **kwargs)
        elif self.encoder_name == "r100":
            self.encoder = iresnet100(False, **kwargs)
        elif self.encoder_name == "r200":
            self.encoder = iresnet200(False, **kwargs)
        elif self.encoder_name == "r2060":
            from .iresnet2060 import iresnet2060
            self.encoder = iresnet2060(False, **kwargs)

        # self.decoder = self.get_decoder_mlp(self.face_embedd_size, self.num_output_points, self.num_axis)
        self.decoder = self.get_decoder_ConvTranspose2d(input_shape=(1, self.face_embedd_size), output_shape=(self.num_axis, self.num_output_points))
        self.classifier = self.get_classifier(self.num_output_points, self.num_axis, num_classes=2)
        self._initialize_weights()


    def get_decoder_ConvTranspose2d(self, input_shape=(1, 256), output_shape=(3, 2500)):
        layers = []

        # layer 1
        k_size = (output_shape[1]-input_shape[1]+1, 1)
        layers.append(nn.ConvTranspose2d(in_channels=1, out_channels=129, kernel_size=k_size, stride=1))
        # layers.append(nn.BatchNorm2d(129, eps=1e-05))
        layers.append(nn.ReLU(True))

        # layer 2
        o_channels = output_shape[0]
        layers.append(nn.ConvTranspose2d(in_channels=129, out_channels=o_channels, kernel_size=(1, 1), stride=1))
        # layers.append(nn.BatchNorm2d(output_shape[0], eps=1e-05))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)


    def get_decoder_mlp(self, input_size=256, num_points=2500, num_axis=3):
        layers = []

        # layer 1
        layers.append(nn.Linear(input_size, num_points, bias=False))
        layers.append(nn.BatchNorm1d(num_points, eps=1e-05))
        layers.append(nn.ReLU(True))

        # layer 2
        layers.append(nn.Linear(num_points, num_points*num_axis, bias=False))
        layers.append(nn.BatchNorm1d(num_points*num_axis, eps=1e-05))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)


    def get_classifier(self, num_points=2500, num_axis=3, num_classes=2):
        layers = []

        # layer 1
        layers.append(nn.Linear(num_points*num_axis, num_classes, bias=False))
        layers.append(nn.BatchNorm1d(num_classes, eps=1e-05))

        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        def _regress_pointcloud(x):
            x = self.encoder(x)              # x.shape = (batch, 256)
            x = x.unsqueeze(1)               # x.shape = (batch, 1, 256)
            x = x.unsqueeze(3)               # x.shape = (batch, 1, 256, 1)

            x = self.decoder(x)              # x.shape = (batch, 3, 2500, 1)
            x = x.squeeze(3)                 # x.shape = (batch, 3, 2500)
            x = torch.permute(x, (0, 2, 1))  # x.shape = (batch, 2500, 3)

            return x

        def _get_logits(x):
            x = x.reshape(x.size(0), self.num_output_points*self.num_axis)
            logits = self.classifier(x)
            return logits

        def _forward(x):
            pred_pc = _regress_pointcloud(x)
            logits = _get_logits(pred_pc)
            return pred_pc, logits

        if self.fp16:
            with torch.cuda.amp.autocast(self.fp16):
                return _forward(x)
        else:
            return _forward(x)
