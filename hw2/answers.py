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

3.



"""

part3_q2 = r"""
Experiment 1.1 shows that larger depth could increase the expressive power of the feature extractor of the model, for instance from 2 layers to 4 we get an improvement in the achieved model accuracy.
Yet, the experiment also shows that for a large number of layers (8 or 16 in that experiment) we might not be able to train the model correctly, as the layers reduced the dimension
of the input to a too low dimension which the linear classifier could not get useful enough information from that latent space, thus the model was untrainable.
"""

part3_q3 = r"""
Experiment 1.2 shows that for sufficient amount of layers 4 in our experiment, 256 filters per layer was optimal which is the maximum number in our experiment,
for a smaller number of layers, the optimal number of filters is smaller, that could be since for smaller number of layers, the output of the convolutional sequence is of
a larger size, for a large number of features that creates a too large output which is harder for the linear calssifier to learn, that is similar to previous experiment as too many layers 
does not allow convergance on the training set, the size of the output of the convolutional part needs not to be too high and not too low.
"""

part3_q4 = r"""
Experiment 1.3 shows that the network converges only for 2 layers, as each layer includes 3 convolutions, thus 2 layers are almost equivalent to 6 layers of single convolution each.
This explains why 4 and 8 does not allow for the linear classifier to be able to converge on the training set.

"""

part3_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q6 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
