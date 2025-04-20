# training-Neural-networks-with-Keras---advnced
# 🧠 Deep Learning for Model Robustness & Keras Customization

This repository presents a collection of hands-on notebooks focusing on improving deep learning model robustness through **regularization techniques** and advanced **Keras customization**, including custom training loops, layers, and losses.

---

## 📁 Part 1: Robust Regularization Techniques in Keras

Explore a wide range of regularization strategies to combat overfitting and enhance model generalization across domains including vision, tabular, and multimodal data.

| Notebook | Description |
|----------|-------------|
| [`01_regularization_variants.ipynb`](https://colab.research.google.com/drive/1J6QuxUER7MO6KnkBDGPJ3Ty7Xo2lyCNd?usp=sharing) | Implements and compares L1, L2, and ElasticNet-style regularization using Keras layers. |
| `https://colab.research.google.com/drive/1QkjK12Ri4xEhoorYTGNsCYo83cUEmCIO?usp=sharing` | Demonstrates dropout techniques within ResNet-like architectures. |
| `03_mc_dropout_uncertainty.ipynb` | Applies Monte Carlo Dropout to quantify uncertainty in predictions. |
| `04_earlystopping_vs_reducelronplateau.ipynb` | Compares EarlyStopping and ReduceLROnPlateau callbacks. |
| `05_weight_init_experiments.ipynb` | Evaluates different initializers: He Normal, Orthogonal, and custom-defined. |
| `06_norm_layers_beyond_batchnorm.ipynb` | Experiments with BatchNorm, LayerNorm, and GroupNorm for stable training. |
| `07_custom_regularizer_from_scratch.ipynb` | Defines and integrates a cosine-based custom regularization function. |
| `08_callbacks_experiment_tracker.ipynb` | Uses WandB/MLflow for real-time experiment tracking and visualization. |
| `09_image_aug_tensorflow_addons.ipynb` | Applies data augmentation using TensorFlow Addons. |
| `10_video_text_augly_augmentation.ipynb` | Uses Meta’s AugLy for augmenting text, video, and images. |
| `11_augment_tabular_data.ipynb` | Demonstrates noise injection and SMOTE-style augmentation for tabular data. |
| `12_fastai_vs_keras_aug.ipynb` | Comparative study of FastAI vs Keras augmentation pipelines. |

🎥 _Full walkthrough video coming soon!_

---

## 📁 Part 2: Advanced Keras – Custom Layers, Losses, Metrics & Training

Deep-dive into the internals of Keras by creating fully customized layers, activations, optimizers, and training loops.

| Notebook | Description |
|----------|-------------|
| `part2_robust_custom_training.ipynb` | Covers OneCycleLR, MC Dropout Alpha, custom normalization layers, advanced logging (WandB), custom loss/metrics, attention blocks (SE/CBAM), contrastive learning, and a flexible training loop implementation. |

---

## ✅ How to Use This Repository

1. **Open any notebook in Google Colab** by clicking the badge or using `Open in Colab` links.
2. **Run all cells sequentially** – dependencies are auto-installed.
3. **Watch the video walkthrough** (where provided) for better understanding.
4. **Fork and remix the notebooks** for your own datasets and experiments.

---

## 📦 Dependencies

- `tensorflow`, `keras`
- `tensorflow-addons`
- `wandb` or `mlflow`
- `augly`
- `scikit-learn`, `matplotlib`, `pandas`
- `fastai` (for comparative augmentation)

Use the following to install everything:

```bash
pip install -r requirements.txt
