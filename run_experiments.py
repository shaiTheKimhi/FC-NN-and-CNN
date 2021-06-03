import hw2.experiments as experiments
from hw2.experiments import load_experiment

"""#experiment 1.1
K = [32, 64]
L = [2,4,8,16]
for k in K:
    for l in L:
        experiments.run_experiment(
            'test_run', seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
            filters_per_layer=[k], layers_per_block=l, pool_every=3, hidden_dims=[100],
            model_type='cnn',
        )


#experiment 1.2
K = [32, 64, 128, 256]
L = [2,4,8]
for k in K:
    for l in L:
        print(f"l:{l}, k:{k}")
        experiments.run_experiment(
            'exp1_2', seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
            filters_per_layer=[k], layers_per_block=l, pool_every=5, hidden_dims=[100],
            model_type='cnn',
        )
"""

#experiment 1.3
k = [64,128,256]
L = [1]#,2,3,4]
for l in L:
    print(f"l:{l}, k:{k}")
    experiments.run_experiment(
        'exp1_3', seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
        filters_per_layer=k, layers_per_block=l, pool_every=5, hidden_dims=[100],padding=1
        model_type='cnn',
    )

"""#experiment 1.4
k = [32]
L = [8,16,32]
for l in L:
    pool = 5 if l < 32 else 10
    experiments.run_experiment(
            'exp1_4', seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
            filters_per_layer=k, layers_per_block=l, pool_every=pool, hidden_dims=[100],
            model_type='resnet',
        )
k = [64,128,256]
L = [2,4,8]
for l in L:
    pool_every = 5 if l < 8 else 8
    experiments.run_experiment(
            'exp1_4', seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
            filters_per_layer=k, layers_per_block=l, pool_every=pool_every, hidden_dims=[100],
            model_type='resnet',
        )
"""

"""#exp2
K = [[64,128,256,512],[32,64,128,256],[64,128,256,128,64],[32],[64],[128],[256]]
L = [2,4,8]
for l in L:
    for k in K:
        pool_every = 5 if l < 8 else 10
        experiments.run_experiment(
                'exp1_4', seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
                filters_per_layer=k, layers_per_block=l, pool_every=pool_every, hidden_dims=[100],
                model_type='ycn',)"""