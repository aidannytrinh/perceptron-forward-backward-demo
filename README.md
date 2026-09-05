# CLIP-Guided RGB Pixel Art: Text-to-Image via Direct Pixel Optimization

A PyTorch notebook that synthesizes an image from a text prompt **without any GAN or VAE** — instead, it directly optimizes the raw RGB pixel values of a multi-resolution image pyramid so that CLIP "sees" the rendered image as matching the prompt, using gradient descent on a CLIP-similarity loss.

> Example prompt used in this run: *"A dragon is flying in the sky on Ho Chi Minh city, Vietnam"*

## Table of Contents

- [Overview](#overview)
- [Why This Is Unusual: Optimizing Pixels, Not Weights](#why-this-is-unusual-optimizing-pixels-not-weights)
- [Requirements](#requirements)
- [Configuration Used in This Run](#configuration-used-in-this-run)
- [Notebook Structure](#notebook-structure)
- [Technical Details](#technical-details)
  - [Multi-Resolution Pyramid Parameterization](#1-multi-resolution-pyramid-parameterization)
  - [YCoCg Color Space + Logit Reparameterization](#2-ycocg-color-space--logit-reparameterization)
  - [CLIP Loss via Random Cutout Augmentation](#3-clip-loss-via-random-cutout-augmentation)
  - [Total Variation (TV) Loss](#4-total-variation-tv-loss)
  - [Two-Stage Optimization Schedule](#5-two-stage-optimization-schedule)
  - [Optimizer Choice](#6-optimizer-choice)
- [How to Run](#how-to-run)
- [Limitations and Notes](#limitations-and-notes)

## Overview

Most "text-to-image" pipelines generate an image by decoding a learned *latent* vector through a pretrained generator (a GAN, VQGAN, or diffusion U-Net) and optimizing that latent so the decoded image matches a CLIP text prompt. This notebook takes a more direct route: the learnable parameters **are the pixels themselves** (organized as a coarse-to-fine pyramid rather than one flat grid), and there is no generator network in the loop at all — only CLIP, used purely as a frozen scoring function.

At every optimization step, the current pixel pyramid is rendered into a full image, several randomly cropped/rotated/jittered "cutouts" of that image are embedded with CLIP, and the loss pushes those embeddings to be more similar (by cosine similarity) to the CLIP text embedding of the prompt. Backpropagating that loss through CLIP and through the pyramid-to-image rendering directly updates the pixel values — no generator weights are ever touched, since CLIP itself is used only in inference mode (`torch.no_grad()`-free only for the loss computation; its own weights are frozen).

## Why This Is Unusual: Optimizing Pixels, Not Weights

If you're used to "training a model" meaning learning generalizable weights from a dataset, this notebook does something different, and it's worth being explicit about it since the notebook is itself titled "...DL_train":

- There is **no dataset** and **no train/val/test split** — the "training" loop optimizes a *single image's* pixel values for a *single, fixed text prompt*.
- The result of running this notebook is **one image**, not a reusable model. To generate a different image, you change the prompt (`texts`) and re-run the entire optimization from scratch (there's no shared, pretrained weight to fine-tune from — this is closer to test-time / inference-time optimization than to conventional supervised training).
- The "model" being optimized has no notion of generalization — overfitting is, in fact, the explicit goal: the pixel pyramid should overfit as precisely as possible to maximizing CLIP similarity with the one prompt it was given.

This category of technique (direct pixel/parameter optimization against a frozen CLIP model) predates and sits alongside diffusion-based text-to-image models; it produces more painterly, abstract-leaning results rather than photorealistic ones, but needs comparatively little GPU memory and no pretrained generator checkpoint.

## Requirements

- A CUDA GPU is strongly recommended (the notebook explicitly targets low-VRAM setups — as little as 4GB — but will be very slow on CPU).
- Python packages:

```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install torch_optimizer
```

`torch_optimizer` supplies the exotic optimizers referenced in the config (`Ranger`, `RAdam`, `AdaBound`, `Yogi`, etc. — see [Optimizer Choice](#6-optimizer-choice)).

## Configuration Used in This Run

| Setting | Value |
|---|---|
| Text prompt | `"A dragon is flying in the sky on Ho Chi Minh city, Vietnam"` (weight 1.0) |
| Image prompts | None |
| Color space | `YCoCg` |
| Aspect ratio | `(4, 3)` |
| Max output dimension | `1000` px |
| Pyramid steps | `11` (plus 1 extra "global color" layer) |
| Pyramid scaling / resample mode | `lanczos` |
| Optimizer | `Ranger` |
| CLIP backbones (ensembled) | `ViT-B/32`, `ViT-B/16` |
| Stage 1 (coarse) | 10,000 cycles, `lr_luma=0.15`, `lr_chroma=0.075`, 2 CLIP cutouts/step |
| Stage 2 (fine) | 1,000 cycles, `lr_luma=0.075`, `lr_chroma=0.0375`, 2 CLIP cutouts/step, LR concentrated on finer pyramid levels |
| Random seed | `None` (non-deterministic — each run produces a different image) |

## Notebook Structure

| Section | Content |
|---|---|
| Setup | Install CLIP + `torch_optimizer`, import libraries, select device |
| Configuration | Text/image prompts, color space, pyramid geometry, optimizer choice, two-stage schedule |
| Pyramid geometry | Compute the pixel dimensions of each pyramid level and the per-level learning-rate scaling |
| Utilities | Image normalization for CLIP, image loading, YCoCg ↔ RGB conversion with logit reparameterization, pyramid → image rendering (Lanczos resample), image display |
| CLIP scoring | Load CLIP model(s) and encode the text prompt(s); `getClipTokens` generates randomly augmented cutouts and embeds them |
| Loss functions | `lossClip` (cosine-similarity loss vs. the text/image prompt embeddings, ensembled across CLIP backbones) and `lossTV` (total variation smoothness regularizer) |
| Resampling internals | `sinc`, `lanczos`, `ramp`, `resample` — a from-scratch Lanczos filter used to combine pyramid levels at arbitrary resolutions |
| Optimization loop | `init_optim` builds per-pyramid-level parameter groups with individual learning rates; `cycle` runs one optimization step (forward render → CLIP loss + TV loss → backward → optimizer step), with periodic "check-in" prints and image previews; `main` initializes the pixel pyramid (random noise, since no `initial_image` is set) and runs both stages in sequence |

## Technical Details

### 1. Multi-Resolution Pyramid Parameterization

Instead of one grid of `H×W` learnable pixels, the image is represented as a stack of **11 layers at increasing resolution** (plus an optional 1×1 "global average color" layer), spaced geometrically between a small starting size and the `max_dim` target — the notebook's `pyramid_lacunarity` is exactly the per-level growth factor needed to reach `max_dim` in `pyramid_steps` steps. The final image is produced by Lanczos-resampling every level up to the target resolution and summing them.

Why bother? This is the same coarse-to-fine idea behind Laplacian pyramids and progressive-growing GANs: low-resolution layers can only represent broad shapes/color blocks, while high-resolution layers add fine detail on top. Giving each level its **own learning rate** (scaled by `lr_scales`, computed from a persistence ratio) lets the optimization settle on large-scale composition first before fine detail is allowed to move much — which is exactly why Stage 1 uses a shallower `pyramid_lr_min` (small-scale detail matters less initially) while Stage 2 sets it to `1` (all levels equally free to refine).

### 2. YCoCg Color Space + Logit Reparameterization

Rather than optimizing R/G/B directly, each pixel is stored in **YCoCg** (luma + two chroma channels), and each channel is passed through `torch.logit` before being treated as the actual learnable parameter — i.e., the true optimization variable `p` satisfies `channel = sigmoid(p)` implicitly (logit is the inverse of sigmoid). This is a standard reparameterization trick: it keeps the *rendered* pixel value naturally bounded in a sensible range without ever needing a hard `clamp()` (which would zero out gradients wherever the clamp is active), while letting the raw parameter `p` itself range freely over all reals during gradient descent. YCoCg specifically separates brightness (luma) from color (chroma), so `lr_luma` and `lr_chroma` can be tuned independently — useful because CLIP similarity tends to be more sensitive to structural/luminance patterns than to exact hue.

### 3. CLIP Loss via Random Cutout Augmentation

`getClipTokens` doesn't embed the whole rendered image once — for each optimization step it generates several ("cuts") randomly rotated, padded, and cropped patches of the image, adds a little noise, and embeds *each* of those with CLIP. This "cutout" technique (well known in the CLIP-guided-generation community) matters for two reasons: (1) CLIP's own training distribution is natural photos, not synthetic pixel-optimization renders, and gradient descent left unconstrained will happily find adversarial pixel patterns that fool a single global CLIP embedding without looking like anything meaningful to a human; averaging the loss over many random crops makes such degenerate shortcuts much harder to find, since the same "trick" would have to survive rotation, cropping, and padding everywhere in the image. (2) It's a cheap form of data augmentation that gives many effective gradient samples per optimization step instead of one.

The loss itself is `1 - cosine_similarity(text_embedding, cutout_embedding)` for a positively-weighted prompt (or `+cosine_similarity` for a negatively-weighted one, letting a prompt be used to *discourage* certain content) — averaged over cutouts and, notably, **ensembled across two separate CLIP backbones** (`ViT-B/32` and `ViT-B/16`), which further reduces the risk of over-optimizing for one specific model's blind spots.

### 4. Total Variation (TV) Loss

```python
def lossTV(image, strength):
    Y = (image[:,:,1:,:] - image[:,:,:-1,:]).abs().mean()
    X = (image[:,:,:,1:] - image[:,:,:,:-1]).abs().mean()
    return (X + Y) * 0.5 * strength
```

This penalizes large differences between horizontally/vertically adjacent pixels — a standard image-denoising regularizer. Without it, pure CLIP-similarity optimization tends to produce high-frequency noisy textures (since CLIP's own robustness to noise means noisy patterns can still score well); TV loss counterbalances that pressure toward a smoother, more visually coherent result. The `denoise` value in each stage controls its strength.

### 5. Two-Stage Optimization Schedule

- **Stage 1** (10,000 cycles): higher overall learning rate, and `pyramid_lr_min=0.2` skews the per-level learning-rate distribution toward the *coarser* pyramid levels — establishing overall composition, color blocking, and rough shapes matching the prompt.
- **Stage 2** (1,000 cycles): roughly half the learning rate, and `pyramid_lr_min=1` gives every pyramid level (coarse and fine) an equal share of the (now smaller) learning-rate budget — polishing detail without disturbing the large-scale structure Stage 1 already settled on.

### 6. Optimizer Choice

The config comments summarize the notebook author's empirical notes on optimizers from the `torch_optimizer` package: `Ranger` (RAdam + Lookahead, used here) "works great"; `RAdam` alone works "extremely well" but can cause color instability; `Yogi` tends to blur; `AdamW` is described as a solid, basic baseline. This run uses **Ranger**.

## How to Run

1. This notebook needs a GPU runtime — in Google Colab, select **Runtime → Change runtime type → GPU** before running.
2. Run the setup cell to install CLIP and `torch_optimizer`.
3. Edit the `texts` list to set your own prompt(s) (and optionally negative-weighted prompts to discourage specific content), and adjust `aspect_ratio` / `max_dim` for your desired output size.
4. Run all cells. Progress images and the CLIP/TV loss values print every `checkin_interval` cycles (every 100 cycles in this configuration) so you can watch the image evolve.
5. Total runtime for the default 10,000 + 1,000 cycle schedule can be substantial (the notebook's own header warns generation "takes a while") — expect it to run for an extended period depending on the GPU.

## Limitations and Notes

- **No reusable model is produced** — each run optimizes pixels for one prompt; there is no checkpoint to save and reuse for other prompts (see [Why This Is Unusual](#why-this-is-unusual-optimizing-pixels-not-weights)).
- **Non-deterministic by default** — `seed = None`, so re-running produces a different image each time; set `seed` to an integer for reproducible results.
- **Output style is a deliberate trade-off**: direct pixel optimization (as opposed to optimizing a GAN/VAE latent) tends toward more painterly/abstract imagery rather than photorealism, but runs on much less GPU memory and needs no pretrained generator weights.
- **This notebook builds on a known community technique** (CLIP-guided direct pixel/pyramid optimization, in the lineage of the broader "CLIP art" notebook family that popularized cutout augmentation, multi-resolution parameterization, and fancy optimizers like Ranger for this purpose). If this notebook was adapted from a specific public template for the course assignment, add a credit/link to that original source in this README before publishing — that's good practice both academically and for open-source attribution.
- **Compute cost scales with `pyramid_steps`, `max_dim`, and total cycles** — reducing any of these will speed up experimentation at the cost of output resolution/detail.

---

## Suggested repo & file naming

- **Repository name:** `clip-guided-rgb-pixel-art`
- **Notebook file name:** `clip_guided_rgb_pixel_optimization.ipynb` (renaming from `23110074_Trinh_Nhat_Anh_DL_train.ipynb` drops the student-ID/course-submission naming so the file's purpose is clear to anyone browsing the repo)
- **README file name:** `README.md` (place at the repo root so it renders automatically on the repo's front page)
