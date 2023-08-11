"""SwinUNETR with cross attention."""
from typing import MutableMapping, Sequence, Tuple, Union

import torch

from monai.networks import blocks
from monai.networks.nets import swin_unetr

if __name__ == '__main__':
    import sys
    sys.path.append('..')
from models import cross_attention

__all__ = [
    "SwinUNETR",
]

FeaturesDictType = MutableMapping[str, torch.Tensor]


class SwinUNETR(swin_unetr.SwinUNETR):
    """SwinUNETR with cross attention."""

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        *args,
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        spatial_dims: int = 3,
        fusion_depths: Sequence[int] = (2, 2, 2, 2, 2, 2),
        cross_attention_in_origin_view: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            fusion_depths: TODO(yiqing).
            cross_attention_in_origin_view: A bool indicates whether compute cross attention in origin view.
                If not, compute cross attention in the view of the first input.

        """
        super().__init__(img_size,
                         *args,
                         num_heads=num_heads,
                         feature_size=feature_size,
                         norm_name=norm_name,
                         spatial_dims=spatial_dims,
                         drop_rate=drop_rate,
                         attn_drop_rate=attn_drop_rate,
                         **kwargs)

        self.encoder5 = blocks.UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.cross_atte6 = cross_attention.TransFusion(
            hidden_size=feature_size * 16,
            num_layers=fusion_depths[5],
            mlp_dim=feature_size * 32,
            num_heads=num_heads[3],
            dropout_rate=drop_rate,
            atte_dropout_rate=attn_drop_rate,
            roi_size=img_size,
            scale=32,
            cross_attention_in_origin_view=cross_attention_in_origin_view)

    def forward_view_encoder(self, x):
        """Encode features."""
        x_hiddens = self.swinViT(x, self.normalize)
        x_enc0 = self.encoder1(x)
        x_enc1 = self.encoder2(x_hiddens[0])
        x_enc2 = self.encoder3(x_hiddens[1])
        x_enc3 = self.encoder4(x_hiddens[2])
        x_enc4 = self.encoder5(x_hiddens[3])  # xa_hidden[3]
        x_dec4 = self.encoder10(x_hiddens[4])
        return {
            'enc0': x_enc0,
            'enc1': x_enc1,
            'enc2': x_enc2,
            'enc3': x_enc3,
            'enc4': x_enc4,
            'dec4': x_dec4,
        }

    def forward_view_decoder(self, x_encoded: FeaturesDictType) -> torch.Tensor:
        """Decode features."""
        x_dec3 = self.decoder5(x_encoded['dec4'], x_encoded['enc4'])
        x_dec2 = self.decoder4(x_dec3, x_encoded['enc3'])
        x_dec1 = self.decoder3(x_dec2, x_encoded['enc2'])
        x_dec0 = self.decoder2(x_dec1, x_encoded['enc1'])
        x_out = self.decoder1(x_dec0, x_encoded['enc0'])
        x_logits = self.out(x_out)
        return x_logits

    def forward_view_cross_attention(
            self, xa_encoded: FeaturesDictType, xb_encoded: FeaturesDictType,
            views: Sequence[int]) -> Tuple[FeaturesDictType, FeaturesDictType]:
        """Inplace cross attention between views."""
        xa_encoded['dec4'], xb_encoded['dec4'] = self.cross_atte6(
            xa_encoded['dec4'], xb_encoded['dec4'], views)
        return xa_encoded, xb_encoded

    def forward(self, xa: torch.Tensor, xb: torch.Tensor,
                views: Sequence[int]) -> Sequence[torch.Tensor]:
        """Two views forward."""
        xa_encoded = self.forward_view_encoder(xa)
        xb_encoded = self.forward_view_encoder(xb)

        xa_encoded, xb_encoded = self.forward_view_cross_attention(
            xa_encoded, xb_encoded, views)
        return [
            self.forward_view_decoder(val) for val in [xa_encoded, xb_encoded]
        ]

    def no_weight_decay(self):
        """Disable weight_decay on specific weights."""
        nwd = {'swinViT.absolute_pos_embed'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    def group_matcher(self, coarse=False):
        """Layer counting helper, used by timm."""
        return dict(
            stem=r'^swinViT\.absolute_pos_embed|patch_embed',  # stem and embed
            blocks=r'^swinViT\.layers(\d+)\.0' if coarse else [
                (r'^swinViT\.layers(\d+)\.0.downsample', (0,)),
                (r'^swinViT\.layers(\d+)\.0\.\w+\.(\d+)', None),
                (r'^swinViT\.norm', (99999,)),
            ])


if __name__ == '__main__':
    from utils import view_ops

    # torch.Size([16, 32, 32, 80, 96])                                                                                                                                                │
    # torch.Size([16, 64, 32, 40, 48])                                                                                                                                                │
    # torch.Size([16, 128, 32, 20, 24])                                                                                                                                               │
    # torch.Size([16, 256, 16, 10, 12])                                                                                                                                               │
    # torch.Size([16, 320, 8, 5, 6])                                                                                                                                                  │

    device = 5
    # x = torch.randn(1, 1, 64, 64, 64).cuda(device)
    x = torch.randn(1, 31, 64, 64, 64).cuda(device)

    # 2, 4, 8, 16, 32
    # x = torch.randn(1, 1, 32, 80, 96).cuda(device)

    roi_x = x.shape[2]
    roi_y = x.shape[3]
    roi_z = x.shape[4]
    in_channels = x.shape[1]
    out_channels = 2
    feature_size = 48
    dropout_path_rate = 0.0
    use_checkpoint = False
    cross_attention_in_origin_view = True

    model = SwinUNETR(
        img_size=(roi_x, roi_y, roi_z),
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        fusion_depths=(1, 1, 1, 1, 1, 1),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
        cross_attention_in_origin_view=cross_attention_in_origin_view,
    )
    model.cuda(device)

    xs, views = view_ops.permute_rand(x)

    y1, y2 = model(xs[0], xs[1], views)
    print(y1.shape, y2.shape)
