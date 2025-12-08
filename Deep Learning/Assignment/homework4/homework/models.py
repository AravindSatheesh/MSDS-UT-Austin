from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dims=(128, 128, 64),
        dropout: float = 0.1,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = n_track * 4  # each track point has 2 coords; left+right -> 4 per point
        output_dim = n_waypoints * 2  # x,y per waypoint

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        # final output layer (no activation)
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # weight init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Basic checks (optional)
        assert track_left.ndim == 3 and track_right.ndim == 3
        b, n_track, c = track_left.shape
        assert n_track == self.n_track and c == 2

        # Concatenate left/right to shape (b, n_track, 4)
        x = torch.cat([track_left, track_right], dim=2)  # (b, n_track, 4)
        x = x.view(b, -1)  # (b, n_track*4)

        out = self.net(x)  # (b, n_waypoints*2)
        out = out.view(b, self.n_waypoints, 2)
        return out

class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Each lane point has 2 coords; we concatenate left + right -> 4-dim per track point
        self.input_proj = nn.Linear(4, d_model)

        # Learned queries: one per waypoint
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # so shapes are (B, S, E)
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Map each decoded latent to (x, y) waypoint
        self.output_proj = nn.Linear(d_model, 2)

        # Initialize weights
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        assert track_left.ndim == 3 and track_right.ndim == 3
        B, n_track, C = track_left.shape
        assert n_track == self.n_track and C == 2, "Unexpected track shape"

        # (B, n_track, 4)
        x = torch.cat([track_left, track_right], dim=2)

        # (B, n_track, d_model)
        memory = self.input_proj(x)

        # Learned queries: (n_waypoints, d_model) -> (B, n_waypoints, d_model)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Transformer decoder: queries attend over memory
        # output: (B, n_waypoints, d_model)
        decoded = self.decoder(tgt=queries, memory=memory)

        # Map to 2D waypoints
        out = self.output_proj(decoded)  # (B, n_waypoints, 2)

        return out

class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        # For input normalization (image in [0,1])
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Simple CNN backbone: (B, 3, 96, 128) -> (B, 256, 1, 1)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 96x128 -> 48x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 48x64 -> 24x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 24x32 -> 12x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 12x16 -> 6x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B, 256, 1, 1)
        )

        # Head: (B, 256) -> (B, n_waypoints*2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_waypoints * 2),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (B, 3, H, W) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (B, n_waypoints, 2)
        """
        assert image.ndim == 4 and image.shape[1] == 3, "Expected (B, 3, H, W) image"

        # Normalize
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # CNN backbone
        x = self.backbone(x)  # (B, 256, 1, 1)

        # Regression head
        x = self.head(x)  # (B, n_waypoints * 2)

        B = x.shape[0]
        x = x.view(B, self.n_waypoints, 2)

        return x


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        # For input normalization (image in [0,1])
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Simple CNN backbone: (B, 3, 96, 128) -> (B, 256, 1, 1)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 96x128 -> 48x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 48x64 -> 24x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 24x32 -> 12x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 12x16 -> 6x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B, 256, 1, 1)
        )

        # Head: (B, 256) -> (B, n_waypoints*2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_waypoints * 2),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        assert image.ndim == 4 and image.shape[1] == 3, "Expected (B, 3, H, W) image"

        # Normalize
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # CNN backbone
        x = self.backbone(x)  # (B, 256, 1, 1)

        # Regression head
        x = self.head(x)  # (B, n_waypoints * 2)

        B = x.shape[0]
        x = x.view(B, self.n_waypoints, 2)

        return x


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
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
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
