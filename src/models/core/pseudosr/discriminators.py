import torch
import torch.nn as nn
import torchvision as TV
import functools

def weight_init_D(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight, 0.1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif class_name.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, scale_factor=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, n_group=1):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        self.n_layers = n_layers
        self.strides, self.n_down = self.comp_strides(scale_factor, n_layers)

        kw = 4
        padw = 0
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=self.strides[0], padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=self.strides[n], padding=padw, bias=use_bias),
                nn.GroupNorm(n_group, ndf*nf_mult) if norm_layer==nn.GroupNorm else norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            nn.GroupNorm(n_group, ndf*nf_mult) if norm_layer==nn.GroupNorm else norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        #self.apply(weight_init_D)

    def comp_strides(self, scale_factor, n_layers):
        strides = [1 for _ in range(n_layers)]
        assert scale_factor in [1, 2, 4]
        n_down = 0
        while True:
            scale_factor = scale_factor // 2
            if scale_factor <= 0:
                break
            n_down += 1
        for s in range(n_down):
            strides[s] = 2
        return strides, n_down

    def forward(self, x):
        """Standard forward."""
        return self.model(x)


if __name__ == "__main__":
    model = NLayerDiscriminator(3, ndf=64, scale_factor=1, n_layers=3, norm_layer=nn.Identity)
    X = torch.randn(2, 3, 32, 32)
    Y = model(X)
    print(model)
    print(Y.shape)
