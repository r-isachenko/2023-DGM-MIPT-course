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
| 1 | September, 5 | <b>Lecture 1:</b> Logistics. Generative models overview and motivation. Problem statement. Divergence minimization framework. Autoregressive models (PixelCNN). | [slides](lectures/lecture1/Lecture1.pdf) |  |
|  |  | <b>Seminar 1:</b> Introduction. Maximum likelihood estimation. Histograms. Kernel density estimation (KDE). | [notebook](seminars/seminar1/seminar1.ipynb) |  |
| 2 | September, 12 | <b>Lecture 2:</b> Bayesian Framework. Latent Variable Models (LVM). Variational lower bound (ELBO). |  |  |
|  |  | <b>Seminar 2:</b> MADE theory and practice. PixelCNN implementation hints. Gaussian MADE. |  |  |
| 3 |  | <b>Lecture 3:</b> EM-algorithm, amortized inference. ELBO gradients, reparametrization trick. Variational Autoencoder (VAE). |  |  |
|  |  | <b>Seminar 3:</b> Latent Variable Models. Gaussian Mixture Model (GMM). GMM and MLE. ELBO and EM-algorithm. GMM via EM-algorithm. |  |  |
| 4 |  | <b>Lecture 4:</b> VAE limitations. Posterior collapse and decoder weakening. Tighter ELBO (IWAE). Normalizing flows prerequisities.  |  |  |
|  |  | <b>Seminar 4:</b> VAE implementation hints. IWAE theory. |  |  |
| 5 |  | <b>Lecture 5:</b> Normalizing Flow (NF) intuition and definition. Forward and reverse KL divergence for NF. Linear flows. |  |  |
|  |  | <b>Seminar 5:</b> Flows. Planar flows. Forward KL vs Reverse KL. Planar flows via Forward KL and Reverse KL. |  |  |
| 6 |  | <b>Lecture 6:</b> Autoregressive flows (gausian AR NF/inverse gaussian AR NF). Coupling layer (RealNVP). NF as VAE model. |  |  |
|  |  | <b>Seminar 6:</b> RealNVP implementation hints. Integer Discrete Flows |  |  |
| 7 |  | <b>Lecture 7:</b> Discrete data vs continuous model. Model discretization (PixelCNN++). Data dequantization: uniform and variational (Flow++). ELBO surgery and optimal VAE prior. Flow-based VAE prior. |  |  |
|  |  | <b>Seminar 7:</b>  Discretization of continuous distribution (MADE++). Aggregated posterior distribution in VAE. VAE with learnable prior. |  |  |
| 8 |  | <b>Lecture 8:</b> Flows-based VAE posterior vs flow-based VAE prior. Likelihood-free learning. GAN optimality theorem. |  |  |
|  |  | <b>Seminar 8:</b> Glow implementation. Vanilla GAN in 1D coding. |  |  |
| 9 |  | <b>Lecture 9:</b> Vanishing gradients and mode collapse, KL vs JS divergences. Adversarial Variational Bayes. Wasserstein distance. Wasserstein GAN (WGAN). |  |  |
|  |  | <b>Seminar 9:</b> KL vs JS divergences. Mode collapse. Vanilla GAN on multimodal 1D and 2D data. Wasserstein distance theory. |  |  |
| 10 |  | <b>Lecture 10:</b> WGAN with gradient penalty (WGAN-GP). Spectral Normalization GAN (SNGAN). f-divergence minimization. GAN evaluation. |  |  |
|  |  | <b>Seminar 10:</b> WGANs on multimodal 2D data. GANs zoo. Evolution of GANs. StyleGAN implementation. |  |  |
| 11 |  | <b>Lecture 11:</b> GAN evaluation (Inception score, FID, Precision-Recall, truncation trick). Discrete VAE latent representations. |  |  |
|  |  | <b>Seminar 11:</b> StyleGAN coding and assessing. Unpaired I2I translation. CycleGAN: discussion and coding. |  |  |
| 12 |  | <b>Lecture 12:</b> Vector quantization, straight-through gradient estimation (VQ-VAE). Gumbel-softmax trick (DALL-E). Neural ODE.  |  |  |
|  |  | <b>Seminar 12:</b> Beyond GANs: Neural Optimal Transport: theory and practice. VQ-VAE implementation hints. |  |  |
| 13 |  | <b>Lecture 13:</b> Adjoint method. Continuous-in-time NF (FFJORD, Hutchinson's trace estimator). Kolmogorov-Fokker-Planck equation and Langevin dynamic. SDE basics. |  |  |
|  |  | <b>Seminar 13:</b> CNF theory. Langevin Dynamics. Energy-based Models. |  |  |
| 14 |  | <b>Lecture 14:</b> Score matching. Noise conditioned score network (NCSN). Gaussian diffusion process. |  |  |
|  |  | <b>Lecture 15:</b> Denoising diffusion probabilistic model (DDPM): objective, link to VAE and score matching. |  |  |

## Homeworks
| Homework | Date | Deadline | Description | Link |
|---------|------|-------------|--------|-------|
| 1 | TBA | TBA | TBA | TBA |
| 2 | TBA | TBA | TBA | TBA |
| 3 | TBA | TBA | TBA | TBA |
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

