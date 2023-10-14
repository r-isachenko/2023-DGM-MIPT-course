# Deep Generative Models course, MIPT, 2023

## Description
The course is devoted to modern generative models (mostly in the application to computer vision).

We will study the following types of generative models:
- autoregressive models,
- latent variable models,
- normalization flow models,
- adversarial models,
- diffusion models.

Special attention is paid to the properties of various classes of generative models, their interrelationships, theoretical prerequisites and methods of quality assessment.

The aim of the course is to introduce the student to widely used advanced methods of deep learning.

The course is accompanied by practical tasks that allow you to understand the principles of the considered models.

## Contact the author to join the course or for any other questions :)

- **telegram:** [@roman_isachenko](https://t.me/roman_isachenko)
- **e-mail:** roman.isachenko@phystech.edu

## Materials

| # | Date | Description | Slides | Video |
|---|---|---|---|---|
| 1 | September, 5 | <b>Lecture 1:</b> Logistics. Generative models overview and motivation. Problem statement. Divergence minimization framework. Autoregressive models (PixelCNN). | [slides](lectures/lecture1/Lecture1.pdf) | [video](https://youtu.be/n9dsiRqkXb8) |
|  |  | <b>Seminar 1:</b> Introduction. Maximum likelihood estimation. Histograms. Kernel density estimation (KDE). | [notebook](seminars/seminar1/seminar1.ipynb) | [video](https://youtu.be/Py9DNGqR7l8) |
| 2 | September, 12 | <b>Lecture 2:</b> Bayesian Framework. Latent Variable Models (LVM). Variational lower bound (ELBO). EM-algorithm, amortized inference. | [slides](lectures/lecture2/Lecture2.pdf) | [video](https://youtu.be/W239stDfszY) |
|  |  | <b>Seminar 2:</b> PixelCNN for MNIST and Binarized MNIST coding. | [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2023-DGM-MIPT-course/blob/main/seminars/seminar2/seminar2.ipynb)<br>[notebook](seminars/seminar2/seminar2.ipynb)<br>[notebook_solved](seminars/seminar2/seminar2_solved.ipynb) | [video](https://youtu.be/vN_qkKce8fg) |
| 3 | September, 19 | <b>Lecture 3:</b> ELBO gradients, reparametrization trick. Variational Autoencoder (VAE). VAE limitations. Tighter ELBO (IWAE).  | [slides](lectures/lecture3/Lecture3.pdf) | [video](https://youtu.be/RqYwaSBrsZc) |
|  |  | <b>Seminar 3:</b> Latent Variable Models. Gaussian Mixture Model (GMM). GMM and MLE. ELBO and EM-algorithm. GMM via EM-algorithm. | [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2023-DGM-MIPT-course/blob/main/seminars/seminar3/seminar3.ipynb)<br>[notebook](seminars/seminar3/seminar3.ipynb) | [video](https://youtu.be/tZMMVgXV3xY) |
| 4 | September, 26 | <b>Lecture 4:</b> Normalizing Flow (NF) intuition and definition. Forward and reverse KL divergence for NF. Linear NF. Gaussian autoregressive NF. | [slides](lectures/lecture4/Lecture4.pdf) | [video](https://youtu.be/0hQiTFT5MyU) |
|  |  | <b>Seminar 4:</b> Variational EM algorithm for GMM. VAE: Implementation hints + Vanilla 2D VAE coding.  | [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2023-DGM-MIPT-course/blob/main/seminars/seminar4/seminar4.ipynb)<br>[notebook](seminars/seminar4/seminar4.ipynb)<br>[notebook_solved](seminars/seminar4/seminar4_solution.ipynb) | [video](https://youtu.be/EIJALHTAVrY) |
| 5 | October, 3 | <b>Lecture 5:</b> Coupling layer (RealNVP). NF as VAE model. Discrete data vs continuous model. Model discretization (PixelCNN++). Data dequantization: uniform and variational (Flow++). | [slides](lectures/lecture5/Lecture5.pdf) | [video](https://www.youtube.com/watch?v=wBb_K51erBE) | |
|  |  | <b>Seminar 5:</b> VAE: posterior collapse, KL-annealing, free-bits. Normalizing flows: basics, planar flows, forward and backward kl for planar flows. | [posterior_collapse](seminars/seminar5/posterior_collapse.ipynb) | [video](https://www.youtube.com/watch?v=8FPAirzmGbA) |
| 6 | October, 10 | <b>Lecture 6:</b> ELBO surgery and optimal VAE prior. NF-based VAE prior. Discrete VAE latent representations. Vector quantization, straight-through gradient estimation (VQ-VAE). | [slides](lectures/lecture6/Lecture6.pdf) | [video](https://youtu.be/7VhdOxWpEQo) |
|  |  | <b>Seminar 6:</b> TBA |  | [video](https://youtu.be/FI3NZt1ZADs) |
| 7 | October, 17 | <b>Lecture 7:</b> Gumbel-softmax trick (DALL-E). Likelihood-free learning. GAN optimality theorem. Adversarial Variational Bayes. | [slides](lectures/lecture7/Lecture7.pdf) |  |
|  |  | <b>Seminar 7:</b>  TBA |  |  |
| 8 |  | <b>Lecture 8:</b> TBA |  |  |
|  |  | <b>Seminar 8:</b> TBA |  |  |
| 9 |  | <b>Lecture 9:</b> TBA |  |  |
|  |  | <b>Seminar 9:</b> TBA |  |  |
| 10 |  | <b>Lecture 10:</b> TBA |  |  |
|  |  | <b>Seminar 10:</b> TBA |  |  |
| 11 |  | <b>Lecture 11:</b> TBA |  |  |
|  |  | <b>Seminar 11:</b> TBA |  |  |
| 12 |  | <b>Lecture 12:</b> TBA |  |  |
|  |  | <b>Seminar 12:</b> TBA |  |  |
| 13 |  | <b>Lecture 13:</b> TBA |  |  |
|  |  | <b>Seminar 13:</b> TBA |  |  |
| 14 |  | <b>Lecture 14:</b> TBA |  |  |
|  |  | <b>Seminar 14:</b> TBA |  |  |

## Homeworks
| Homework | Date | Deadline | Description | Link |
|---------|------|-------------|--------|-------|
| 1 | September, 13 | September, 27 | <ol><li>Theory (KDE, alpha-divergences, curse of dimensionality).</li><li>PixelCNN (receptive field, autocomplete) on MNIST.</li><li>ImageGPT on MNIST.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw1.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2023-DGM-MIPT-course/blob/main/homeworks/hw1.ipynb) |
| 2 | September, 27 | October, 11 | <ol><li>Theory (IWAE theory, EM-algorithm for GMM).</li><li>VAE on CIFAR10.</li><li>ResNetVAE on CIFAR10.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw2.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2023-DGM-MIPT-course/blob/main/homeworks/hw2.ipynb) |
| 3 | October, 11 | October, 25 | <ol><li>Theory (Sylvester flows, NF expressivity, Discrete vs Continuous).</li><li>RealNVP on 2D data.</li><li>RealNVP on CIFAR10.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw3.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2023-DGM-MIPT-course/blob/main/homeworks/hw3.ipynb) |
| 4 | TBA | TBA | TBA | TBA |
| 5 | TBA | TBA | TBA | TBA |
| 6 | TBA | TBA | TBA | TBA |

## Game rules
- 6 homeworks each of 13 points = **78 points**
- oral cozy exam = **26 points**
- maximum points: 78 + 26 = **104 points**
### Final grade: `floor(relu(#points/8 - 2))`

## Prerequisities
- probability theory + statistics
- machine learning + basics of deep learning
- python + basics of one of DL frameworks (pytorch/tensorflow/etc)

## Previous episodes
- [2022-2023, autumn-spring, MIPT](https://github.com/r-isachenko/2022-2023-DGM-MIPT-course)
- [2022, autumn, AIMasters](https://github.com/r-isachenko/2022-2023-DGM-AIMasters-course)
- [2022, spring, OzonMasters](https://github.com/r-isachenko/2022-DGM-Ozon-course)
- [2021, autumn, MIPT](https://github.com/r-isachenko/2021-DGM-MIPT-course)
- [2021, spring, OzonMasters](https://github.com/r-isachenko/2021-DGM-Ozon-course)
- [2020, autumn, MIPT](https://github.com/r-isachenko/2020-DGM-MIPT-course)

