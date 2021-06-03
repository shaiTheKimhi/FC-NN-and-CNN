import torch
import torch.nn as nn
import itertools as it
from typing import Sequence

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
            self,
            in_size,
            out_classes: int,
            channels: Sequence[int],
            pool_every: int,
            hidden_dims: Sequence[int],
            conv_params: dict = {},
            activation_type: str = "relu",
            activation_params: dict = {},
            pooling_type: str = "max",
            pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        """
        note: it was easier to implement the logic of self._n_features()
        in here, since we already iterate over the self.channels, thus
        i simply calculated the h,w and channels of each layer, and return the last one.
        """
        # initiate as defult params:
        '''padding, stride, kernel_size, pool_kernel_size = 0, 1, 3, 2'''
        #load params if needed:
        names = ['padding', 'stride', 'kernel_size', 'Pkernel_size']
        values = [0, 1, 3, 2]
        for k in names:
            if type(self.conv_params) is dict and k in self.conv_params.keys():
                values[names.index(k)] = self.conv_params[k]
            if type(self.pooling_params) is dict and k[1:] in self.pooling_params.keys():
                values[names.index(k)] = self.pooling_params[k[1:]]
        padding,stride,kernel_size,pool_kernel_size = tuple(values)


        #define an update the output_size regards to conv and pooling functions
        update_conv_size = lambda x: int((x + 2 * padding - kernel_size) / stride) + 1
        update_pool_size = lambda x: int((x - pool_kernel_size) / pool_kernel_size) + 1

        for i, out_channels in enumerate(self.channels):
            #conv
            layers.append(nn.Conv2d(in_channels,out_channels, kernel_size,stride,padding)) #optional- add dilation
            # activision
            active_layer = nn.ReLU() if self.activation_type == "relu" else nn.LeakyReLU(**self.activation_params)
            layers.append(active_layer)
            # update feature size
            in_h, in_w = update_conv_size(in_h), update_conv_size(in_w)
            # pooling
            if (i + 1) % self.pool_every == 0:
                pool_layer = nn.MaxPool2d(pool_kernel_size) if self.pooling_type == "max" else nn.AvgPool2d(pool_kernel_size)
                layers.append(pool_layer)
                # update feature size
                in_h,in_w = update_pool_size(in_h),update_pool_size(in_w)

            in_channels = out_channels
        self.n_features = in_channels * in_h * in_w

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            return self.n_features
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_classifier(self):
        layers = []

        # Discover the number of features after the CNN part.
        n_features = self._n_features()

        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        num_of_features = self._n_features()
        for dim in self.hidden_dims:

            layers.append(nn.Linear(num_of_features, dim))
            num_of_features = dim
            active_layer = nn.ReLU() if self.activation_type == "relu" else nn.LeakyReLU(**self.activation_params)
            layers.append(active_layer)
        layers.append(nn.Linear(num_of_features, self.out_classes))
        # ========================

        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x) #forward conv
        features = features.reshape(features.size(0), -1) #flatten
        out = self.classifier(features) #classifier (usually FC)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
            self,
            in_channels: int,
            channels: Sequence[int],
            kernel_sizes: Sequence[int],
            batchnorm: bool = False,
            dropout: float = 0.0,
            activation_type: str = "relu",
            activation_params: dict = {},
            **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        main_layers = []

        curr_in_channels = in_channels

        # block layers
        for i, channel in enumerate(channels):
            main_layers.append(nn.Conv2d(in_channels=curr_in_channels, out_channels=channel,
                                         kernel_size=kernel_sizes[i], padding=int((kernel_sizes[i] - 1) / 2)))
            if i == len(channels) - 1:
                break
            if dropout > 0:
                main_layers.append(nn.Dropout2d(dropout))

            if batchnorm:
                main_layers.append(nn.BatchNorm2d(channel))
            active_layer = nn.ReLU() if activation_type == "relu" else nn.LeakyReLU(**activation_params)
            main_layers.append(active_layer)

            curr_in_channels = channel

        shortcut = nn.Sequential()
        # skip connection using 1x1 kernel
        if in_channels != channels[-1]:
            shortcut = nn.Sequential(nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False))

        self.main_path = nn.Sequential(*main_layers)
        self.shortcut_path = shortcut
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
            self,
            in_out_channels: int,
            inner_channels: Sequence[int],
            inner_kernel_sizes: Sequence[int],
            **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        # ====== YOUR CODE: ======
        channels = [inner_channels[0]] + list(inner_channels)
        channels.append(in_out_channels)
        kernel_size = [1] + list(inner_kernel_sizes)
        kernel_size.append(1)

        super().__init__(in_out_channels, channels, kernel_size, **kwargs)
        # ========================


class ResNetClassifier(ConvClassifier):
    def __init__(
            self,
            in_size,
            out_classes,
            channels,
            pool_every,
            hidden_dims,
            batchnorm=False,
            dropout=0.0,
            **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======
        num_blocks = int(len(self.channels) / self.pool_every)
        update_pool_size = lambda x: int((x - 2) / 2) + 1 #pool kernel size set to 2

        p = self.pool_every
        n = len(self.channels)
        for i in range(num_blocks):
            layers.append(ResidualBlock(in_channels, self.channels[i*p : (i+1)*p] , [3]*p, self.batchnorm, self.dropout, self.activation_type, self.activation_params))
            if self.pooling_type == "avg":
                pool_layer = nn.MaxPool2d(2) if self.pooling_type == "max" else nn.AvgPool2d(2)  ##pool kernel size set to 2
                layers.append(pool_layer)
                in_h, in_w = update_pool_size(in_h), update_pool_size(in_w)
            in_channels = self.channels[(i+1)*p - 1]

        if n%p != 0:
            layers.append(ResidualBlock(in_channels, self.channels[-(n%p):], [3]*(n%p) , self.batchnorm, self.dropout, self.activation_type, self.activation_params))
            in_channels = self.channels[-1]

        self.n_features = in_channels * in_h * in_w
        # ========================
        seq = nn.Sequential(*layers)
        return seq

class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes,*args, **kwargs):
        """
        See ConvClassifier.__init__
        """
        super().__init__(in_size, out_classes, *args, **kwargs)

        # TODO: Add any additional initialization as needed.
        # ====== YOUR CODE: ======
        self.batchnorm = True
        # ========================

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        self.batchnorm = True
        self.dropout = 0.3
        self.activation_type = "relu"
        
        self.activation_params["negative_slope"] = 0.05
        print((self.activation_params))
        
        
        '''
            curr_channels = channels[(len(channels) - len(channels)%self.pool_every):]
            '''
        #######
        pool_size = self.pooling_params.get("kernel_size", 2)# if "kernel_size" in self.pooling.params.keys() else 2
        pool_pad = self.pooling_params.get("padding", 0)
        pool_stride = self.pooling_params.get("stride", pool_size)
        ###
        
        num_blocks = int(len(self.channels) / self.pool_every)
        update_pool_size = lambda x: int((x - pool_size) / pool_size) + 1 #pool kernel size set to 2

        p = self.pool_every
        n = len(self.channels)
        for i in range(num_blocks):
            layers.append(ResidualBlock(in_channels, self.channels[i*p : (i+1)*p] , [3]*p, self.batchnorm, self.dropout, self.activation_type, self.activation_params))
            
            pool_layer = nn.MaxPool2d(pool_size) if self.pooling_type == "max" else nn.AvgPool2d(pool_size)  ##pool kernel size set to 2
            layers.append(pool_layer)
            in_h, in_w = update_pool_size(in_h), update_pool_size(in_w)
            in_channels = self.channels[(i+1)*p - 1]

        if n%p != 0:
            layers.append(ResidualBlock(in_channels, self.channels[-(n%p):], [3]*(n%p) , self.batchnorm, self.dropout, self.activation_type, self.activation_params))
            in_channels = self.channels[-1]

        self.n_features = in_channels * in_h * in_w
        # ========================
        seq = nn.Sequential(*layers)
        return seq
    # ========================