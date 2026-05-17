```{raw} typst
#set page(margin: auto)
```

# **Evaluation & Discussion**

All metrics are computed on the official MNIST test set (10 000 samples, never seen during training or validation). Each of the 25 runs uses identical hyperparameters — only the architecture and normalization strategy vary. Results are reported as test accuracy, weighted F1, and Matthews Correlation Coefficient (MCC).

## **Full Grid Ranking**

The results split cleanly by architecture, not by normalization. MiniResNet leads all 25 runs (99.38–99.58%), with the best single result being MiniResNet trained on augmented data at 99.58% test accuracy. DeepCNN and DeepDNN cluster tightly in the high-98% range, and LinearDNN sits a full percentage point lower — a gap that no normalization strategy can close. The choice of how to scale the inputs barely moves the needle for any of these four; the ceiling is set by the architecture itself.

SimpleCNN is the exception. Without BatchNorm, its performance swings nearly 20 points depending solely on input scale — from 96.15% with raw pixels down to 76.87% with min-max normalization. Every other architecture stays within 2 points across all five strategies. That contrast is the clearest finding in the entire grid: normalization matters only when the architecture gives it room to matter.

Accuracy, weighted F1, and MCC rank all 25 runs in essentially the same order, confirming that on a balanced ten-class dataset, any of the three is a reliable summary of model quality.

```{figure} figures/test_accuracy.png
:name: fig-accuracy
:width: 75%
Test accuracy (%) for all 25 model–normalization combinations.
```

```{figure} figures/weighted_f1.png
:name: fig-wf1
:width: 75%
Weighted F1 (%) for all 25 model–normalization combinations.
```

```{figure} figures/mcc.png
:name: fig-mcc
:width: 75%
MCC (×100) for all 25 model–normalization combinations.
```

```{raw} typst
#set page(margin: auto)
```

## **Model Comparison**

| Model | Params | Avg Acc | Best | Best Norm | Worst | Worst Norm | Spread |
|---|---:|---:|---:|---|---:|---|---:|
| MiniResNet | 112,394 | 99.48% | 99.58% | augmented | 99.38% | symmetric | 0.20% |
| DeepCNN | 160,330 | 98.86% | 99.02% | augmented | 98.71% | none | 0.31% |
| DeepDNN | 569,226 | 98.49% | 98.54% | augmented | 98.47% | minmax | 0.07% |
| LinearDNN | 101,770 | 96.43% | 97.46% | standardize | 95.57% | symmetric | 1.89% |
| SimpleCNN | 28,426 | 84.59% | 96.15% | none | 76.87% | minmax | 19.28% |

Four of the five architectures include BatchNorm, and for those four the **Spread** column is almost irrelevant — swapping normalization strategies moves accuracy by at most 0.31 points. The normalization layer inside the network corrects for input scale before activations reach the deeper layers, so it barely matters how the pixels arrive. LinearDNN is the outlier among them at 1.89 points, but even that is small compared to what actually separates the models from each other.

### SimpleCNN anomaly

SimpleCNN is in a different category entirely. Its 19-point swing between `none` (96.15%) and `minmax` (76.87%) is not a normalization problem — it is a BatchNorm problem. When min-max scaling compresses pixel values to [0, 1], the filters in the first convolutional layer receive signals too small to respond to meaningfully. Raw [0, 255] values restore enough activation magnitude for the filters to fire. Without BatchNorm to compensate, the model is fully exposed to whatever the input pipeline hands it — and the confusion matrix below shows what that costs: digits 2, 4, 5, and 9 all fall below 75% F1, with classes that should be separable bleeding into one another because the features that distinguish them never get activated.

```{figure} figures/simplecnn.png
:name: fig-simplecnn
:width: 105%
SimpleCNN (minmax) — the worst run (76.87%).
```

```{raw} typst
#set page(margin: auto)
```

## **Normalization Comparison**

| Norm | Avg Acc (4 models) | Best single result |
|---|---:|---:|
| standardize | 98.59% | 99.56% (MiniResNet) |
| augmented | 98.55% | 99.58% (MiniResNet) |
| minmax | 98.24% | 99.48% (MiniResNet) |
| none | 98.13% | 99.39% (MiniResNet) |
| symmetric | 98.07% | 99.38% (MiniResNet) |

SimpleCNN is excluded here — its BatchNorm-driven anomaly would dominate the averages and obscure any signal from normalization itself.

Among the four BatchNorm architectures, the differences between strategies are small. Standardization edges out the rest on average (98.59%) because using the exact dataset statistics ($\mu = 0.1307$, $\sigma = 0.3081$) centres activations in the optimal range for gradient flow. Augmentation produces the single best result (99.58%, MiniResNet) by forcing the model to generalise across rotations, translations, and scale variations — but it pulls down the average because it actively hurts LinearDNN. LinearDNN_standardize (97.46%) outperforms LinearDNN_augmented (97.05%): geometric perturbations are only useful to a model that understands spatial structure, and one that flattens all pixels into a vector cannot exploit them. The gap between the top and bottom strategy is just 0.52 points — on architectures with BatchNorm, normalization is a secondary concern.

---

## **Efficiency**

| Model | Params | Avg train/run (s) | Avg inference (ms/sample) | Checkpoint (KB) | Best Acc |
|---|---:|---:|---:|---:|---:|
| SimpleCNN | 28,426 | 951 | 0.239 | 115 | 96.15% |
| LinearDNN | 101,770 | 838 | 0.009 | 400 | 97.46% |
| MiniResNet | 112,394 | 2,259 | 0.672 | 456 | 99.58% |
| DeepCNN | 160,330 | 994 | 0.339 | 637 | 99.02% |
| DeepDNN | 569,226 | 718 | 0.034 | 2,238 | 98.54% |

The table compares models on three dimensions averaged across all five normalization runs: parameter count, wall-clock training time, and inference speed. Parameter count and checkpoint size are fixed per architecture regardless of normalization.

MiniResNet is the most parameter-efficient model — with 112k parameters it outperforms DeepCNN (160k, 99.02%) and DeepDNN (569k, 98.54%), the latter using 5× the parameters for 1% lower accuracy. Residual connections allow deep gradient flow without stacking extra layers, and Global Average Pooling replaces a large dense classifier with a single value per channel. The scatter plot below puts this in perspective: MiniResNet sits at the Pareto frontier of the accuracy–efficiency trade-off, achieving the highest accuracy at a lower parameter cost than both heavier models.

```{figure} figures/efficiency_scatter.png
:name: fig-efficiency
:width: 75%
Test accuracy vs parameter count for each architecture's best run.
```

The one area where MiniResNet pays a real cost is training time (~2 259 s/run on CPU), driven by its two residual blocks. LinearDNN sits at the opposite extreme — 75× faster at inference — but is capped at 97.46% by its inability to exploit spatial structure. For latency-critical applications where that accuracy ceiling is acceptable, LinearDNN is the practical choice.

The learning curves tell the same story. DeepDNN and MiniResNet reach near-ceiling accuracy within the first five epochs and show no overfitting, while LinearDNN converges more slowly and plateaus below the CNN models. SimpleCNN_none converges but at a much lower ceiling — the limit is not training duration but the absence of regularization. Within the BatchNorm architectures, normalization has no meaningful effect on convergence speed: all five strategies reach their respective ceilings within the same epoch range for a given architecture.

```{figure} figures/learning_curves1.png
:name: fig-curves1
:width: 105%
Learning curves for LinearDNN (standardize), DeepDNN (augmented), and SimpleCNN (none).
```

```{figure} figures/learning_curves2.png
:name: fig-curves2
Learning curves for DeepCNN (augmented) and MiniResNet (augmented).
```

---

## **Per-digit Analysis**

Two patterns hold across every architecture. Digit 1 is consistently the easiest — its narrow vertical stroke is distinctive enough that even a model with no spatial awareness can separate it reliably. Digit 5 is consistently the hardest, most often confused with 6 and 9 due to shared curved strokes; it has the lowest recall in the best-performing model and the widest error spread in the weakest one.

SimpleCNN breaks the pattern in a revealing way. While its digit 1 F1 stays near 99%, digits 2, 5, and 9 fall below 95% — a spread far larger than any other architecture shows. Without BatchNorm, activations vary wildly depending on stroke thickness and pixel density, so the model happens to learn some digits well and fails systematically on others. It is not a difficulty problem — it is a normalization problem expressed at the per-class level.

```{figure} figures/per_digit_f1.png
:name: fig-per-digit
Per-digit F1 (%) for each architecture's best run.
```

```{raw} typst
#set page(margin: auto)
```

## **Best Model — MiniResNet augmented**

MiniResNet trained on augmented data is the strongest configuration across all 25 runs: 99.58% test accuracy, MCC = 0.9953, macro and weighted F1 both at 0.9957–0.9958 — the negligible difference between the two is expected on a balanced dataset. More telling than the aggregate numbers is the per-class breakdown: every single digit exceeds 99% F1, and the confusion matrix is nearly diagonal.

```{figure} figures/miniresnet.png
:name: fig-best
:width: 105%
MiniResNet (augmented) — Confusion matrix, per-class precision/recall/F1, and learning curves.
```

---

# **Conclusion**

This project trained five neural network architectures under five input normalization strategies — 25 runs in total — to isolate how much each factor independently affects MNIST digit classification accuracy. The controlled setup makes the answer unusually clear: architecture is the dominant factor, and normalization is secondary.

The spread between the best and worst architecture (MiniResNet at 99.58% vs. LinearDNN at 97.46%) is 2.12 points. The largest normalization-induced swing within a single architecture is 1.89 points — and that is LinearDNN, an outlier. For every CNN architecture, swapping normalization strategies moves accuracy by under 0.35 points. The ceiling is set by the architecture, not by how pixels are scaled.

The one exception that proves the rule is SimpleCNN. Its 19-point swing between `none` and `minmax` is not a normalization effect — it is what happens when BatchNorm is absent. Without it, the network has no internal mechanism to correct for input scale, so the preprocessing pipeline becomes the dominant factor. Every architecture with BatchNorm is effectively immune to this; SimpleCNN is fully exposed to it.

On the efficiency side, MiniResNet achieves the highest accuracy with just 112k parameters — outperforming DeepDNN at one-fifth the size. Residual connections give gradients a direct path to early layers, removing the need for brute-force depth. Among normalization strategies, standardization has the best average while augmentation produces the single best result; for spatially-aware models, augmentation is the stronger choice.

The recommended configuration is MiniResNet with augmented normalization: 99.58% test accuracy, MCC = 0.9953, no digit below 99% F1, and compact enough to run at interactive speeds as demonstrated in the digit recognition demo.

# **References**

Gorodkin, J. (2004). Comparing two K-category assignments by a K-category correlation coefficient. *Computational Biology and Chemistry*, 28(5–6), 367–374.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 448–456.

Johansson, H. (2020). *MNIST Dataset* [Data set]. Kaggle.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.