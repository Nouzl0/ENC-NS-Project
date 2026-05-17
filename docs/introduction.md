# **Introduction**

Neural network performance depends on two largely independent decisions: the choice of architecture and the choice of input preprocessing. In practice these are often tuned together, making it difficult to attribute accuracy gains to either factor in isolation. This project separates the two by training five architectures — ranging from a single-layer linear network to a residual network — under five input normalization strategies, producing a fully controlled 5×5 grid of 25 independent runs on the MNIST handwritten digit dataset. All runs share identical hyperparameters, so any difference in outcome can be attributed directly to architecture or normalization rather than to training configuration.

## **Research Questions**

1. Does input normalization affect final accuracy and convergence, and if so, by how much relative to architecture choice?
2. Which architecture achieves the best accuracy-to-parameter ratio on MNIST?

## **Goals**

1. Implement five neural network architectures spanning the linear-to-residual design space.
2. Train each architecture under five input normalization strategies, producing 25 controlled runs.
3. Evaluate every run on accuracy, macro and weighted F1, MCC, parameter count, training time, and inference speed.
4. Identify which architecture–normalization combination generalizes best and at what computational cost.
5. Deploy the best-performing model in an interactive demo that accepts hand-drawn digits and returns live predictions.

```{raw} typst
#set page(margin: auto)
```

## **Dataset**

MNIST (*Modified National Institute of Standards and Technology*) (LeCun et al., 1998) is the standard benchmark for handwritten digit classification. The dataset version used in this project is sourced from Johansson (2020).

| Property | Value |
|---|---|
| Source | [Kaggle — hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) |
| Total samples | 70 000 grayscale images |
| Image size | 28 × 28 px, 1 channel |
| Classes | 10 (digits 0–9) |
| Train split | 48 000 (80 % of official train set) |
| Validation split | 12 000 (20 % of official train set) |
| Test split | 10 000 (official test set, never seen during training) |

```{figure} figures/mnist_dataset.png
:name: fig-mnist
:width: 60%
MNIST training set
```

---

## **Experiment at a Glance**

The experiment forms a complete 5×5 grid — every architecture trained under every normalization strategy. No combination is skipped, ensuring a fair cross-comparison with no cherry-picking.

| | LinearDNN | DeepDNN | SimpleCNN | DeepCNN | MiniResNet |
|---|:-:|:-:|:-:|:-:|:-:|
| none | ● | ● | ● | ● | ● |
| minmax | ● | ● | ● | ● | ● |
| standardize | ● | ● | ● | ● | ● |
| symmetric | ● | ● | ● | ● | ● |
| augmented | ● | ● | ● | ● | ● |

Each ● represents one independent training run — 25 in total. All runs use the same optimizer, learning rate, batch size, and number of epochs, so any difference in outcome is attributable solely to architecture or normalization.