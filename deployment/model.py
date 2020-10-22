from fastai.vision.all import *
from fastai.vision.learner import _default_meta
from fastai.vision.models.unet import _get_sz_change_idxs, UnetBlock, ResizeToOrig


class DynamicUnetDIY(SequentialEx):
    "Create a U-Net from a given architecture."

    def __init__(
        self,
        arch=resnet50,
        n_classes=32,
        img_size=(96, 128),
        blur=False,
        blur_final=True,
        y_range=None,
        last_cross=True,
        bottle=False,
        init=nn.init.kaiming_normal_,
        norm_type=None,
        self_attention=None,
        act_cls=defaults.activation,
        n_in=3,
        cut=None,
        **kwargs
    ):
        meta = model_meta.get(arch, _default_meta)
        encoder = create_body(
            arch, n_in, pretrained=False, cut=ifnone(cut, meta["cut"])
        )
        imsize = img_size

        sizes = model_sizes(encoder, size=imsize)
        sz_chg_idxs = list(reversed(_get_sz_change_idxs(sizes)))
        self.sfs = hook_outputs([encoder[i] for i in sz_chg_idxs], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        ni = sizes[-1][1]
        middle_conv = nn.Sequential(
            ConvLayer(ni, ni * 2, act_cls=act_cls, norm_type=norm_type, **kwargs),
            ConvLayer(ni * 2, ni, act_cls=act_cls, norm_type=norm_type, **kwargs),
        ).eval()
        x = middle_conv(x)
        layers = [encoder, BatchNorm(ni), nn.ReLU(), middle_conv]

        for i, idx in enumerate(sz_chg_idxs):
            not_final = i != len(sz_chg_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sizes[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sz_chg_idxs) - 3)
            unet_block = UnetBlock(
                up_in_c,
                x_in_c,
                self.sfs[i],
                final_div=not_final,
                blur=do_blur,
                self_attention=sa,
                act_cls=act_cls,
                init=init,
                norm_type=norm_type,
                **kwargs
            ).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sizes[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, act_cls=act_cls, norm_type=norm_type))
        layers.append(ResizeToOrig())
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(
                ResBlock(
                    1,
                    ni,
                    ni // 2 if bottle else ni,
                    act_cls=act_cls,
                    norm_type=norm_type,
                    **kwargs
                )
            )
        layers += [
            ConvLayer(ni, n_classes, ks=1, act_cls=None, norm_type=norm_type, **kwargs)
        ]
        apply_init(nn.Sequential(layers[3], layers[-2]), init)
        # apply_init(nn.Sequential(layers[2]), init)
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"):
            self.sfs.remove()
