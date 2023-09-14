# Comparison methods hyperparameters

## Graphsage
```
k: 2
Samples in layer 1: 25
Samples in layer 2: 10
Output dim 1: 128
Output dim 2: 128
Learning rate: 1e-5
Dropout: 0
Weight decay: 0
Max degree: 100
```

## ECC
```
Learning rate: 5e-3
Weight decay: 5e-4
Momentum: 0.9
Model: i_0.1_0.2,c_16,b,r,d_0.3,m_1e10_1e10,b,r,f_14 =>
    in sequence
        Initial resolution: 0.1
        Initial radius: 0.2
        Graph convolution: 16
        Batch normalization -> eps: 1e-5
        Relu: true
        Dropout: 0.3
        Pooling: max
Out res: 1e10
Out radius: 1e10
Batch normalization -> eps: 1e-5
	Relu: true
	Fully connected layer:
		Output features: 14
	Point cloud processing:
		Dropout: 0.1
		Augmentation scale: 1.1
		Augmentation random rotation around z-axis: 1
		Augmentation probability of mirroring about x or y axes: 0.5
		Neighborhood building: radius-search
		Edge attribute definition: polar coordinates
```

## DGCNN

```
Sort pooling k: 0.6 - 0.9
Final dense layer’s hidden size: 128
Learning rate: 1e-4 - 1e-5
```

## DiffPool

```
Learning rate: 1e-3
Hidden dimension: 64
Embedding dimension: 64
Pool ratio: 0.25
Activation: relu
```

## GIN

```
Hidden units: 16, 32, 64
Dropout: 0, 0.5
```
## sGIN

```
Hidden units: 16, 32, 64
Dropout: 0, 0.5
```
## KerGNN

```
Kernel: drw, rw
Learning rate: 1e-2, 1e-3
K: 1, 2, 3
Subgraph size: 8, 10, 16
Hidden dimensions: [16, 32]
Mlp layers: 1, 5, 9
Mlp hidden dimensions: 16, 32, 64
Size graph filter: 6
Max step: 1, 2
Dropout: 0, 0.4, 0.5
```
## RWGNN

```
Learning rate: 1e-3
Hidden graphs: 8
Hidden graph nodes: 6
Random walk P: 3
```
## WL

```
Data set size: 10, 100, 1000
Graph size: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
Subtree height: 4, 2, 8
Graph density: 0.4, 0.1 to 0.9 +0.1
```
