import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from models.unet_parts import *
#from models.conv_sam import *


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 and 'SAM' not in classname or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        # print(net)
        net.to(gpu_ids[0])
        #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    #init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf=64, netG='att', norm='batch', use_dropout=False, 
    init_type='normal', init_gain=0.02, gpu_ids=[], n_att=5, argmax=False):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'our':
        net = ResnetGenerator_our(input_nc, output_nc, ngf, n_blocks=9)
    elif netG == 'att':
        net = ResnetGenerator_Att(input_nc, output_nc, ngf, n_blocks=9, n_attentions=n_att, argmax=argmax)
    elif netG == 'unet':
        net = UNet(input_nc, output_nc)
    elif netG == 'unet_att':
        net = UNet_Att(input_nc, output_nc, n_attentions=n_att)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)
    #return net


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).to(prediction.device)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetGenerator_Att(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, n_attentions=5, argmax=False):
        super(ResnetGenerator_Att, self).__init__()
        self.argmax = argmax
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = n_blocks
        self.conv1 = nn.Conv2d(input_nc, ngf, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)
        self.n_att = n_attentions

        self.resnet_blocks = []
        for i in range(n_blocks):
            self.resnet_blocks.append(resnet_block(ngf * 4, 3, 1, 1))
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.deconv1_content = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.deconv1_norm_content = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_content = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm_content = nn.InstanceNorm2d(ngf)
        self.deconv3_content = nn.Conv2d(ngf, self.n_att*self.output_nc, 7, 1, 0)

        self.deconv1_attention = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.deconv1_norm_attention = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_attention = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm_attention = nn.InstanceNorm2d(ngf)
        self.deconv3_attention = nn.Conv2d(ngf, self.n_att, 1, 1, 0)
        
        self.tanh = torch.nn.Tanh()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks(x)
        x_content = F.relu(self.deconv1_norm_content(self.deconv1_content(x)))
        x_content = F.relu(self.deconv2_norm_content(self.deconv2_content(x_content)))
        x_content = F.pad(x_content, (3, 3, 3, 3), 'reflect')
        content = self.deconv3_content(x_content)
        # Extract the content
        image = self.tanh(content)
        images = []
        for i in range(self.n_att-1):
            images.append(image[:, i*self.output_nc : (i+1)*self.output_nc, :, :])

        x_attention = F.relu(self.deconv1_norm_attention(self.deconv1_attention(x)))
        x_attention = F.relu(self.deconv2_norm_attention(self.deconv2_attention(x_attention)))

        attention = self.deconv3_attention(x_attention)

        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)
        if self.argmax:
            values, indices = attention.max(dim=1, keepdim=True)
            attention = torch.zeros(attention.shape).to(attention.device).scatter(1, indices, values)
        attentions = []
        for i in range(self.n_att):
            attentions.append(attention[:, i:(i+1), :, :])

        outputs = []
        output = 0
        for i in range(self.n_att):
            if i < self.n_att - 1:
                out = images[i] * attentions[i]
            else:
                out = input * attentions[i]
            output += out
            outputs.append(out)

        return output, outputs, attentions, images



class ResnetGenerator_our(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(ResnetGenerator_our, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = n_blocks
        self.conv1 = nn.Conv2d(input_nc, ngf, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)

        self.resnet_blocks = []
        for i in range(n_blocks):
            self.resnet_blocks.append(resnet_block(ngf * 4, 3, 1, 1))
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        # self.resnet_blocks1 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks1.weight_init(0, 0.02)
        # self.resnet_blocks2 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks2.weight_init(0, 0.02)
        # self.resnet_blocks3 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks3.weight_init(0, 0.02)
        # self.resnet_blocks4 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks4.weight_init(0, 0.02)
        # self.resnet_blocks5 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks5.weight_init(0, 0.02)
        # self.resnet_blocks6 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks6.weight_init(0, 0.02)
        # self.resnet_blocks7 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks7.weight_init(0, 0.02)
        # self.resnet_blocks8 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks8.weight_init(0, 0.02)
        # self.resnet_blocks9 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks9.weight_init(0, 0.02)

        self.deconv1_content = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.deconv1_norm_content = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_content = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm_content = nn.InstanceNorm2d(ngf)
        self.deconv3_content = nn.Conv2d(ngf, 9*self.output_nc, 7, 1, 0)

        self.deconv1_attention = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.deconv1_norm_attention = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_attention = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm_attention = nn.InstanceNorm2d(ngf)
        self.deconv3_attention = nn.Conv2d(ngf, 10, 1, 1, 0)
        
        self.tanh = torch.nn.Tanh()
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks(x)
        # x = self.resnet_blocks1(x)
        # x = self.resnet_blocks2(x)
        # x = self.resnet_blocks3(x)
        # x = self.resnet_blocks4(x)
        # x = self.resnet_blocks5(x)
        # x = self.resnet_blocks6(x)
        # x = self.resnet_blocks7(x)
        # x = self.resnet_blocks8(x)
        # x = self.resnet_blocks9(x)
        x_content = F.relu(self.deconv1_norm_content(self.deconv1_content(x)))
        x_content = F.relu(self.deconv2_norm_content(self.deconv2_content(x_content)))
        x_content = F.pad(x_content, (3, 3, 3, 3), 'reflect')
        content = self.deconv3_content(x_content)
        # Extract the content
        image = self.tanh(content)
        image1 = image[:, 0:1*self.output_nc, :, :]
        # print(image1.size()) # [1, 3, 256, 256]
        image2 = image[:, 1*self.output_nc:2*self.output_nc, :, :]
        image3 = image[:, 2*self.output_nc:3*self.output_nc, :, :]
        image4 = image[:, 3*self.output_nc:4*self.output_nc, :, :]
        image5 = image[:, 4*self.output_nc:5*self.output_nc, :, :]
        image6 = image[:, 5*self.output_nc:6*self.output_nc, :, :]
        image7 = image[:, 6*self.output_nc:7*self.output_nc, :, :]
        image8 = image[:, 7*self.output_nc:8*self.output_nc, :, :]
        image9 = image[:, 8*self.output_nc:9*self.output_nc, :, :]
        # image10 = image[:, 27:30, :, :]

        x_attention = F.relu(self.deconv1_norm_attention(self.deconv1_attention(x)))
        x_attention = F.relu(self.deconv2_norm_attention(self.deconv2_attention(x_attention)))
        # x_attention = F.pad(x_attention, (3, 3, 3, 3), 'reflect')
        # print(x_attention.size()) [1, 64, 256, 256]
        attention = self.deconv3_attention(x_attention)

        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)

        attention1_ = attention[:, 0:1, :, :]
        attention2_ = attention[:, 1:2, :, :]
        attention3_ = attention[:, 2:3, :, :]
        attention4_ = attention[:, 3:4, :, :]
        attention5_ = attention[:, 4:5, :, :]
        attention6_ = attention[:, 5:6, :, :]
        attention7_ = attention[:, 6:7, :, :]
        attention8_ = attention[:, 7:8, :, :]
        attention9_ = attention[:, 8:9, :, :]
        attention10_ = attention[:, 9:10, :, :]

        attention1 = attention1_.repeat(1, self.output_nc, 1, 1)
        # print(attention1.size())
        attention2 = attention2_.repeat(1, self.output_nc, 1, 1)
        attention3 = attention3_.repeat(1, self.output_nc, 1, 1)
        attention4 = attention4_.repeat(1, self.output_nc, 1, 1)
        attention5 = attention5_.repeat(1, self.output_nc, 1, 1)
        attention6 = attention6_.repeat(1, self.output_nc, 1, 1)
        attention7 = attention7_.repeat(1, self.output_nc, 1, 1)
        attention8 = attention8_.repeat(1, self.output_nc, 1, 1)
        attention9 = attention9_.repeat(1, self.output_nc, 1, 1)
        attention10 = attention10_.repeat(1, self.output_nc, 1, 1)

        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        output8 = image8 * attention8
        output9 = image9 * attention9
        # output10 = image10 * attention10
        output10 = input * attention10

        o = output1 + output2 + output3 + output4 + output5 + output6 + output7 + output8 + output9 + output10
        return o, output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, attention1,attention2,attention3, attention4, attention5, attention6, attention7, attention8,attention9,attention10, image1, image2,image3,image4,image5,image6,image7,image8,image9

# resnet block with reflect padding
class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))

        return input + x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UNet_Att(nn.Module):
    def __init__(self, input_nc, output_nc, n_attentions = 5, num_downs=7, bilinear=False):
        super(UNet_Att, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        self.n_middle = num_downs-4

        self.inc = DoubleConv(self.input_nc, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down_mid = nn.ModuleList([Down(512,512) for _ in range(self.n_middle)])
        factor = 2 if bilinear else 1

        self.down4 = Down(512, 1024 // factor)

        self.up0 = Up(1024, 512 // factor, bilinear, inner_most=True)
        ### Content

        self.cont_up_mid = nn.ModuleList([Up(512, 512, bilinear) for _ in range(self.n_middle)])
        self.cont_up1 = Up(512, 256 // factor, bilinear)
        self.cont_up2 = Up(256, 128 // factor, bilinear)
        self.cont_up3 = Up(128, 64, bilinear)
        self.cont_up4 = Up(64, 64, bilinear)
        self.cont_outc = OutConv(64, (n_attentions-1)*self.output_nc)

        ### Attention
        #self.att_up_mid = nn.ModuleList([Up(512, 512, bilinear) for _ in range(self.n_middle)])
        self.att_up1 = Up(512, 256 // factor, bilinear)
        self.att_up2 = Up(256, 128 // factor, bilinear)
        self.att_up3 = Up(128, 64, bilinear)
        self.att_up4 = Up(64, 64, bilinear)
        self.att_outc = OutConv(64, n_attentions*self.output_nc)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        middle = []
        for down in self.down_mid:
            middle.append(down(x4))
        middle = middle[::-1]
        x5 = self.down4(middle[0])

        x0 = self.up0(x5, None)

        for i, up in enumerate(self.cont_up_mid):
            x = up(x0, middle[i])
        x = self.cont_up1(x, x4)
        x = self.cont_up2(x, x3)
        x = self.cont_up3(x, x2)
        x = self.cont_up4(x, x1)
        images = self.cont_outc(x)

        att = self.att_up1(att, x4)
        att = self.att_up2(att, x3)
        att = self.att_up3(att, x2)
        att = self.att_up4(att, x1)
        att = self.att_outc(att)
        ## TODO

        return None

class UNet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=7, bilinear=False):
        super(UNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        self.n_middle = num_downs-4

        self.inc = DoubleConv(self.input_nc, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down_mid = nn.ModuleList([Down(512,512) for _ in range(self.n_middle)])
        factor = 2 if bilinear else 1

        self.down4 = Down(512, 1024 // factor)

        ### Content
        self.cont_up1 = Up(1024, 512 // factor, bilinear, inner_most=True)
        self.cont_up_mid = nn.ModuleList([Up(512, 512, bilinear) for _ in range(self.n_middle)])
        self.cont_up2 = Up(512, 256 // factor, bilinear)
        self.cont_up3 = Up(256, 128 // factor, bilinear)
        self.cont_up4 = Up(128, 64, bilinear)
        self.cont_up5 = Up(64, 64, bilinear)
        self.cont_outc = OutConv(64, self.output_nc)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        middle = []
        for down in self.down_mid:
            middle.append(down(x4))
        middle = middle[::-1]
        x5 = self.down4(middle[0])

        x = self.cont_up1(x5, None)

        for i, up in enumerate(self.cont_up_mid):
            x = up(x, middle[i])
        x = self.cont_up2(x, x4)
        x = self.cont_up3(x, x3)
        x = self.cont_up4(x, x2)
        x = self.cont_up5(x, x1)
        logits = self.cont_outc(x)
        return logits


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
        
def Cor_CoeLoss(y_pred, y_target):
    x = y_pred
    y = y_target
    x_var = x - torch.mean(x)
    y_var = y - torch.mean(y)
    r_num = torch.sum(x_var * y_var)
    r_den = torch.sqrt(torch.sum(x_var ** 2)) * torch.sqrt(torch.sum(y_var ** 2))
    r = r_num / r_den

    # return 1 - r  # best are 0
    return 1 - r**2 # abslute constrain

if __name__ == '__main__':
    gpu_ids = [0, 1]
    g = define_G(input_nc=1, output_nc=1, netG='conv_sam', gpu_ids=gpu_ids)
    #g = UNet(input_nc=1, output_nc=1)
    x = torch.rand(3, 1, 256, 256).cuda()
    output = g(x)
    print(output.shape)