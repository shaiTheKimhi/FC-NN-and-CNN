r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. First, let's denote the FC  layer as $Z = X W^T + b$ and the Jacobian tensor denotes by $\Delta Z$, that is simply W.
since batch size come usually first, the size of W (the Jacobian tensor) is [N, in_features,out_features]
in our case `[128 , 1024, 2048]`

2. The number of parameter of the jacobian w.r.t the input is $128 \times 1024 \times 2048$.
float32 means 32 bits, and each byte equal to 8 bits,
means $128 \times 1024 \times 2048 \times 32 / 8 = 1073741824 bytes $ that is about 1.07 GB.
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.05
    lr = 0.02
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.05
    lr_vanilla = 0.05
    lr_momentum = 0.005#25
    lr_rmsprop = 0.0001
    reg = 0.01
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,



    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.5
    lr = 0.005
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
1. Dropout is a regularization method that should generalize better (reduce generalization error)
by reducing the complexity of the model.
thus, we expected the model withoud dropout to overfit, and that is what we see as the accuracy of the train goes up while the test stay low.
The 40% dropout reach lower test loss and looks like indeed overfit less
2. The higher dropout rate of 80% seems to act like too much of regularization and converge to a high loss.
this risk of under fitting exist with every regularization that is too strong on the model.
"""

part2_q2 = r"""
It is possible.
The accuracy measures count correct samples while the loss measures of the score (probability-like) of how far the correct answer.
We could "be less certain" about some samples, but still to have the highest score.
(uncertainty usually measure the softmax output of the correct class)
or, we could be really wrong in some samples but to achive more correct ones.

For example,for classes (denote by 0,1,2 etc), lest look at example when $Y= 0$ and the softmax output is [0.4,0.6,0...0].
that means that $\^{Y} = 1 $, while we can update the weights to have softmax output of [0.3,0.1,0.1,...0] (of course that should sum to 1)
in that case $\^{Y} = 0 $ so we increase accuracy measure, but loss would be -log(0.3) instead of -log(0.4)
(please note that the example lake sigmas and talk about one example just to demonstrate the notion)
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""

1. Number of parameters: <br\> 
Regular block: 2 conv layers, each one has kernel size 3X3 with 256 input channels and 256 output channels thus, number of parameters is $2\times 3^2\times256^2$ = `1,179,648`
Bottleneck block: first convolution has 1X1 kernel from 256 channels to 64, second kernel has 3X3 kernel with input 64 and output 64 channels and last convolution has 1X1 kernel from 64 channels to 256. Thus, total number of parameters is $256\times 64 + 3^2 \times 64^2 + 64\times 256=$ `69,632`.
Bottleneck has much fewer parameters

2.Number of Operations: <br\>
Let N be input image size ($W\times H$)
Regular blocks: for each 3X3 convolution number of operations is $OP=N\times 256\times (3^2+1) \times 256$ total number of operations is $2\times OP+N\times 256=$ `1,310,976N`
note: we count the summary of each kernel multiplication as one operation.
Bottleneck block: 1X1 convolution takes $O1=N\times 256\times (1^2+1)\times 64$ opreations, 3X3 convolution takes $O2=N\times 64\times (3^2+1) \times 64$ total number of operations is: $2\times O1+O2+256N$ = `102,656N`

3. The bottleneck receptive field is $3\times3$, while the regular block has 2 $3\times3$ convs, thus the receptive field is $5\times5$.
Across feature maps,both uses all input channels, and that is why both combine the channels  onto a given spatial area.
"""

part3_q2 = r"""
Experiment 1.1 shows that larger depth **could increase** the expressive power of the feature extractor of the model (from 2 layers to 4, we get an improvement of accuracy).
Yet, the experiment also shows that for a large number of layers (8 or 16 in that experiment) **could decrease** the accuracy.
That phenomanan could happen for several reasons, One might be that each conv layer reduced the dimension of the input, and the features become too small to learn from.
Second reason could be that a network that is too long, suffer from vanishing gradients, since it has more layers to go throw.
Another reason could be that after several layers, we already extracted the right features, and more conv layers "ruined" the features, since as we learned in the tutorials, learning the identity function without residual might be hard.
To fix that problem, we can use skip connections, as we use in residual, that could solve the vanishing gradients for instanse.
We can also use padding to keep the features of each layer at the same size to avoid the first mentioned problem (or use less pooling layers).
"""

part3_q3 = r"""
Experiment 1.2 shows that for sufficient amount of layers (4, corrolated with exp 1.1), 256 filters per layer preformed optimally,
for 2 layers, the optimal number of filters is smaller, 
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx FILL LATER
that could be since for smaller number of layers, the output of the convolutional sequence is of
a larger size, for a large number of features that creates a too large output which is harder for the linear calssifier to learn, that is similar to previous experiment as too many layers 
does not allow convergance on the training set, the size of the output of the convolutional part needs not to be too high and not too low.
"""

part3_q4 = r"""
Experiment 1.3 creats an architecture of L layers 3 times (2 for each k in K),i.e, we get 3L conv layers.
We already saw in previus experiments that cnn network does not preform well for this problem with deep architecture (as sugested in q3.2).
2 phenomanas that we see is:
1. L=2 get good results (since the network is not too deep).
2. increasing the number of features gradualy, works better then a direct channel lifting (i.e- 3-64-128-256 works better then 3-256-256-256).
"""

part3_q5 = r"""
First thing that is obvios to mention, is that Resnet architecture, that use the skip connections, solve part of the vanishing gradients problem we mentioned in 1.1.
for that reason, we can use deeper networks and from the results of 1.2, deeper networks can use more features.
We still notice that too deep of a network does not learn the best, and L=4 for the K=[64,128,256] preform the best. but that could point the first problem we mentioned of feature size
(when using less pooling and padding for same size convs, the deeper model with L=8 can outpreform  the L=4).
in general, we see that the total accuracy is higher (regardless every analysis of the graph phenomanons).


"""

part3_q6 = r"""
Our model is based on Resnet with several architecture changes, dropout of 0.4, relu activisions and expiroments with **Gradual ascent** and **Gradual ascent-gradual descent** in the number of channels.
We tried to combine all that we learned from the previus models in this HW: in the manner of model complexity, non-vanishing gradients, number of features and feature size (less pooling_every since we use deeper models).
We also tried diffrent activision functions but with no much sucsess to beat the Relu.
**Gradual ascent-gradual descent** architecture is an idea in the notion of bottleneck and worked pretty well for us (better then using bottelneck residual as is).
Since the last part of exp 1 got a fine result, our result is not much better then that and achive ~79 percent, as opposed to 75 in part 1.4/ 
The idea in some of the expirements was to imitate the resnet18 architecture.
Note that in the SOTA papers on this dataset, all use data augmentations and we think that our result don't progress much since we didn't change the data.
"""
# ==============
