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
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = 2 * n_track * 2  # left and right, each with (n_track, 2)
        output_dim = n_waypoints * 2

        hidden_dim1 = 256
        hidden_dim2 = 256

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim2, output_dim),
        )

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
        # Flatten and concatenate left/right boundary points
        batch_size = track_left.shape[0]
        left_flat = track_left.reshape(batch_size, -1)
        right_flat = track_right.reshape(batch_size, -1)
        x = torch.cat([left_flat, right_flat], dim=1)

        preds = self.mlp(x)
        preds = preds.view(batch_size, self.n_waypoints, 2)

        return preds


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.d_model = d_model
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Encode 2D points to d_model
        self.point_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # Side embedding: 0 -> left, 1 -> right
        self.side_embed = nn.Embedding(2, d_model)

        # Positional index embedding along the boundary polyline
        self.pos_embed = nn.Embedding(n_track, d_model)

        # Transformer decoder (queries attend to encoded boundary points)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            batch_first=False,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Projection from decoded query features to 2D waypoint
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2),
        )

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
        batch_size = track_left.shape[0]

        # Prepare memory (boundary features): concatenate left and right along sequence dimension
        # Shapes: (B, n_track, 2) each -> (B, 2*n_track, 2)
        memory_points = torch.cat([track_left, track_right], dim=1)

        # Encode points (B, S, 2) -> (B, S, d_model)
        memory_feats = self.point_encoder(memory_points)

        # Add side and positional embeddings
        # Positional indices 0..n_track-1 for each side
        pos_indices = torch.arange(self.n_track, device=memory_points.device)
        pos_embed_left = self.pos_embed(pos_indices)  # (n_track, d)
        pos_embed_right = self.pos_embed(pos_indices)  # (n_track, d)
        pos_embed_both = torch.cat([pos_embed_left, pos_embed_right], dim=0)  # (2*n_track, d)
        pos_embed_both = pos_embed_both.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 2*n_track, d)

        # Side ids: 0 for left, 1 for right
        side_ids = torch.cat([
            torch.zeros(self.n_track, dtype=torch.long, device=memory_points.device),
            torch.ones(self.n_track, dtype=torch.long, device=memory_points.device),
        ], dim=0)
        side_embed_both = self.side_embed(side_ids)  # (2*n_track, d)
        side_embed_both = side_embed_both.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 2*n_track, d)

        memory_feats = memory_feats + pos_embed_both + side_embed_both  # (B, S, d)

        # Transformer expects (S, B, d)
        memory = memory_feats.transpose(0, 1)  # (S=2*n_track, B, d)

        # Queries: learned embeddings repeated for batch, shape (T=n_waypoints, B, d)
        queries = self.query_embed.weight.unsqueeze(1).expand(-1, batch_size, -1)

        decoded = self.decoder(tgt=queries, memory=memory)  # (T, B, d)
        decoded = decoded.transpose(0, 1)  # (B, T, d)

        waypoints = self.output_proj(decoded)  # (B, T, 2)
        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Lightweight CNN backbone
        def conv_block(in_channels: int, out_channels: int, stride: int = 1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.backbone = nn.Sequential(
            conv_block(3, 32, stride=2),   # 48x64
            conv_block(32, 64, stride=2),  # 24x32
            conv_block(64, 128, stride=2), # 12x16
            conv_block(128, 128, stride=2),# 6x8
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        x = self.backbone(x)
        x = self.global_pool(x)
        x = self.head(x)
        x = x.view(x.shape[0], self.n_waypoints, 2)

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
