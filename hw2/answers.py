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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
