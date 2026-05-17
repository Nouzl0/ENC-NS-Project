```{raw} typst
#set page(margin: auto)
```

# **Analysis**
This chapter covers the theoretical foundations of the experiment — the dataset, problem formulation, normalization strategies, neural network building blocks, and the evaluation metrics used to compare all 25 runs.


## **Dataset Selection**

MNIST (Modified National Institute of Standards and Technology) (LeCun et al., 1998) is a benchmark dataset of 70 000 grayscale images of handwritten digits (0–9), each sized 28×28 pixels. Each image is a single-channel pixel grid with values in $[0, 255]$, where 0 is black and 255 is white. The label is an integer in $\{0, \ldots, 9\}$ corresponding to the written digit. The dataset is pre-divided into 60 000 training images and 10 000 test images. For this project the training set is further split into 48 000 training and 12 000 validation samples using a fixed random seed, ensuring reproducibility across runs.

MNIST was selected as the experimental dataset for three reasons:
- **Controlled complexity** — images are small enough to train dozens of models on CPU without excessive compute, yet complex enough to differentiate architecture quality.
- **Established benchmark** — test accuracy on MNIST is well-documented across decades of literature, making results directly interpretable and comparable to published work.
- **Class balance** — all 10 digit classes are approximately equally represented (~7 000 samples each), which prevents class imbalance from confounding the comparison between architectures and normalization strategies.

---

## **Problem Classification**

Digit recognition is a **supervised multiclass classification** problem. Each sample is a labeled pair $(\mathbf{x}_i, y_i)$, where $\mathbf{x}_i \in \mathbb{R}^{784}$ is the flattened pixel vector and $y_i \in \{0, \ldots, 9\}$ is the ground-truth digit. The model learns a mapping $f: \mathbf{x} \mapsto y$ by minimizing categorical cross-entropy over the training set:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid \mathbf{x}_i)$$

The output layer produces 10 logits $z_0, \ldots, z_9$, one per class. Softmax converts them to a probability distribution, and the predicted class is the highest-probability digit:

$$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=0}^{9} e^{z_j}}, \qquad \hat{y} = \arg\max_k P(y = k \mid \mathbf{x})$$

---

## **Input Normalization**

Raw MNIST pixel values lie in $[0, 255]$ — a large magnitude range that causes disproportionately large gradients in the first layer and slows convergence. Five strategies are evaluated — one unnormalized baseline, three static transforms, and one augmentation-based variant — each implemented as a `(train_transform, eval_transform)` pair. For the first four both transforms are identical; for augmented, geometric perturbations apply only during training while validation and test always use standardization.

```{raw} typst
#pagebreak()
```

### None — Unnormalized Baseline
`ToTensor()` is applied and values are immediately scaled back to $[0, 255]$. This is the negative reference — it establishes what happens without any normalization and makes the benefit of the other strategies measurable.

$$x' = x \cdot 255$$

### Min-Max

`ToTensor()` alone, which divides pixel values by 255 and produces a $[0, 1]$ range:

$$x' = \frac{x}{255}$$

Preserves the original distribution shape. Removes the magnitude problem without shifting or centering the data.

### Standardization (Z-score)

Subtracts the MNIST dataset mean and divides by its standard deviation, producing zero-mean, unit-variance activations:

$$x' = \frac{x - \mu}{\sigma}, \qquad \mu = 0.1307,\quad \sigma = 0.3081$$

These statistics are computed over the full MNIST training set. Centering near zero keeps activations in the most sensitive region of ReLU and sigmoid nonlinearities, which is optimal for gradient flow.

### Symmetric

Normalizes to $[-1, 1]$ using fixed statistics instead of the dataset-derived ones:

$$x' = \frac{x - 0.5}{0.5} = 2x - 1$$

Equivalent to standardization with $\mu = 0.5$, $\sigma = 0.5$. Centered and symmetric, but slightly suboptimal because the fixed parameters do not match the true MNIST distribution.

### Augmented

Applies random geometric perturbations during training to artificially increase sample diversity, then standardizes. Validation and test use standardization only — no augmentation:

| Transform | Parameters |
|---|---|
| `RandomAffine` — rotation | ±10° |
| `RandomAffine` — translation | ±10% of image size |
| `RandomAffine` — scale | 90–110% |
| Val/Test transform | Standardization ($\mu=0.1307$, $\sigma=0.3081$) |

Augmentation forces the model to generalize across handwriting styles, orientations, and scales rather than memorizing exact pixel positions.

```{raw} typst
#pagebreak()
```


## **Neural Network Architectures**

The five architectures in this project are built from four core building blocks — dense layers, convolutional layers, batch normalization, and residual connections. Understanding each component in isolation makes the architectural choices and their trade-offs easier to reason about.

### Fully Connected Networks

A fully connected (dense) layer maps every input feature to every output neuron:

$$\mathbf{y} = \sigma(W\mathbf{x} + \mathbf{b})$$

where $W \in \mathbb{R}^{m \times n}$ is a learned weight matrix and $\sigma$ is a nonlinear activation function. Applied to images, the input is first flattened into a vector, treating each pixel as an independent feature. This discards all spatial structure — the network has no way to know that neighboring pixels are related.

### Convolutional Networks

A convolutional layer slides a small learned filter $\mathbf{w}$ across the spatial dimensions of the input, computing a dot product at each position:

$$(f * w)[i, j] = \sum_{m}\sum_{n} x[i+m,\, j+n] \cdot w[m, n]$$

Each filter detects a local pattern — an edge, a curve, a corner — and produces one feature map (LeCun et al., 1998). Two properties make convolutions far more parameter-efficient than dense layers for image data:

- **Weight sharing** — the same filter is applied at every spatial location, so the number of parameters depends on filter size, not image size.
- **Translation equivariance** — a pattern detected in one part of the image is recognized everywhere else.

### Batch Normalization

Introduced by Ioffe & Szegedy (2015), batch normalization normalizes activations across a mini-batch to zero mean and unit variance, then re-scales them with learned parameters $\gamma$ and $\beta$:

$$\hat{x}i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \varepsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta$$

This prevents activations from drifting to extreme values during training, allowing higher learning rates and faster convergence. It also acts as a mild regularizer. In this project, BatchNorm is used in `DeepDNN`, `DeepCNN`, and `MiniResNet` — but deliberately omitted from `LinearDNN` and `SimpleCNN`, which serve as unnormalized baselines.

### Residual Connections

Deep networks suffer from vanishing gradients — by the time the loss signal propagates back through many layers, it becomes too small to update early weights effectively. Residual connections, introduced in ResNet (He et al., 2016), solve this by adding the block's input directly to its output:

$$\text{output} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

The network learns the residual — the correction to the identity — rather than a full mapping from scratch. The skip connection gives gradients a direct path to earlier layers regardless of depth. This is used exclusively in `MiniResNet`.

### Global Average Pooling

After the convolutional blocks, the feature map has shape $(C, H, W)$ — $C$ channels, each of size $H \times W$. Global Average Pooling collapses each channel to a single scalar by averaging across its spatial extent:

$$\text{GAP}(f)_k = \frac{1}{H \times W} \sum_{i,j} f_k[i, j]$$

This produces a vector of length $C$ that summarizes what the network detected, without caring where it detected it. All three CNN architectures (`SimpleCNN`, `DeepCNN`, `MiniResNet`) use GAP before the final classification layer, which significantly reduces parameter count compared to flattening the full feature map.

---


## **Evaluation Metrics**

All classification metrics are derived from the **confusion matrix** $C$, where $C_{ij}$ is the number of samples with true class $i$ that were predicted as class $j$. The diagonal entries $C_{ii}$ are correct predictions; off-diagonal entries are mistakes. A perfect classifier has a purely diagonal confusion matrix.

### Accuracy
The overall fraction of correct predictions across all 10 classes:

$$\text{Acc} = \frac{\sum_i C_{ii}}{\sum_{i,j} C_{ij}}$$

Reliable when classes are balanced, which holds for MNIST. Per-class accuracy $\text{Acc}_i = C_{ii} / \sum_j C_{ij}$ is also computed for each digit independently, revealing which specific digits the model struggles with most regardless of overall performance.

### Precision, Recall, F1

Accuracy alone hides how errors are distributed. Precision and recall break this down per class:

- **Precision** — of all samples the model predicted as class $i$, how many actually were class $i$? High precision means few false alarms.
- **Recall** — of all samples that truly are class $i$, how many did the model catch? High recall means few misses.

$$\text{Precision}_i = \frac{C_{ii}}{\sum_j C_{ji}}, \qquad \text{Recall}_i = \frac{C_{ii}}{\sum_j C_{ij}}$$

```{raw} typst
#pagebreak()
```

A model can trivially maximize one at the expense of the other — predicting everything as class $i$ gives perfect recall but terrible precision. F1 is their harmonic mean and penalizes this imbalance:

$$F1_i = \frac{2 \cdot \text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$

Both **macro** (unweighted mean across classes) and **weighted** (mean weighted by class support) averages are reported. For balanced datasets like MNIST the two are nearly identical, but macro is more sensitive to rare-class failures.

### Matthews Correlation Coefficient (MCC)

Accuracy and F1 can still look good on a model that systematically confuses two specific classes. MCC avoids this by treating the confusion matrix as a correlation problem — it measures how well the predicted label distribution matches the true one across all classes at once (Gorodkin, 2004):

$$\text{MCC} = \frac{N \sum_k C_{kk} - \sum_k p_k s_k}{\sqrt{\left(N^2 - \sum_k p_k^2\right)\left(N^2 - \sum_k s_k^2\right)}}$$

where $N$ is total samples, $p_k = \sum_j C_{kj}$ is the row sum (all true instances of class $k$), and $s_k = \sum_j C_{jk}$ is the column sum (all predictions of class $k$). The scale is intuitive: MCC = 1 is a perfect classifier, MCC = 0 is no better than random guessing, MCC = −1 is perfectly wrong on every sample. It is the primary scalar used to rank all 25 runs in the results.