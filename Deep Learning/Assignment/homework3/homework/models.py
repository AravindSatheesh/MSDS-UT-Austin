from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

def conv_bn_relu(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )
class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        width: int = 32,
        dropout: float = 0.2,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        C = width
        self.stem = conv_bn_relu(in_channels, C, 3, 1, 1)

        self.block1 = nn.Sequential(
            conv_bn_relu(C, C, 3, 1, 1),
            conv_bn_relu(C, C, 3, 1, 1),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            conv_bn_relu(C, C*2, 3, 1, 1),
            conv_bn_relu(C*2, C*2, 3, 1, 1),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            conv_bn_relu(C*2, C*4, 3, 1, 1),
            conv_bn_relu(C*4, C*4, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),  # -> (B, C*4, 1, 1)
        )
        self.classifier = nn.Linear(C*4, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        z = self.stem(z)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        z = torch.flatten(z, 1)
        logits = self.classifier(z)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)

class UpBlock(nn.Module):
    """ConvTranspose2d upsample + convs (with skip)"""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            conv_bn_relu(out_c + skip_c, out_c, 3, 1, 1),
            conv_bn_relu(out_c, out_c, 3, 1, 1),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # handle odd input sizes by center-cropping skip if needed
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            skip = skip[..., dh//2: skip.size(-2) - (dh - dh//2), dw//2: skip.size(-1) - (dw - dw//2)]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
    
class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        width: int = 16,
        depth_head_act: str = "sigmoid",  # ensures [0,1]
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        C = width
        # Encoder
        self.enc1 = nn.Sequential(conv_bn_relu(in_channels, C, 3, 1, 1),
                                  conv_bn_relu(C, C, 3, 1, 1))
        self.down1 = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)  # /2
        self.enc2 = nn.Sequential(conv_bn_relu(C*2, C*2, 3, 1, 1),
                                  conv_bn_relu(C*2, C*2, 3, 1, 1))
        self.down2 = nn.Conv2d(C*2, C*4, kernel_size=3, stride=2, padding=1)  # /4
        self.enc3 = nn.Sequential(conv_bn_relu(C*4, C*4, 3, 1, 1),
                                  conv_bn_relu(C*4, C*4, 3, 1, 1))

        # Decoder
        self.up1 = UpBlock(C*4, C*2, C*2)  # -> /2
        self.up2 = UpBlock(C*2, C, C)      # -> /1

        # Heads
        self.seg_head = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, num_classes, 1),
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, 1, 1),
        )
        self.depth_head_act = depth_head_act

        # init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder
        e1 = self.enc1(z)              # (B, C, H, W)
        e2 = self.enc2(self.down1(e1)) # (B, 2C, H/2, W/2)
        b  = self.enc3(self.down2(e2)) # (B, 4C, H/4, W/4)

        # Decoder with skips
        d1 = self.up1(b, e2)           # (B, 2C, H/2, W/2)
        d2 = self.up2(d1, e1)          # (B, C, H, W)

        logits = self.seg_head(d2)             # (B, num_classes, H, W)

        depth = self.depth_head(d2)                # (B, 1, H, W)
        if self.depth_head_act == "sigmoid":
            depth = depth.sigmoid()
        raw_depth = depth.squeeze(1)

        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
