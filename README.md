# OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism

<p align="center">
  <a href="https://arxiv.org/abs/2604.14762">
    <img src="https://img.shields.io/badge/arXiv-2604.14762-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv"/>
  </a>
  <a href="poster.pdf">
    <img src="https://img.shields.io/badge/Poster-PDF-1f6feb.svg?logo=adobeacrobatreader&logoColor=white" alt="Poster"/>
  </a>
  <a href="#citation">
    <img src="https://img.shields.io/badge/BibTeX-Cite-success.svg" alt="BibTeX"/>
  </a>
</p>

<p align="center">
  <b>Jordan Shipard</b><sup>1,2</sup>&nbsp;&nbsp;
  Arnold Wiliem<sup>1,2</sup>&nbsp;&nbsp;
  Kien Nguyen Thanh<sup>2</sup>&nbsp;&nbsp;
  Wei Xiang<sup>3</sup>&nbsp;&nbsp;
  Clinton Fookes<sup>2</sup>
  <br/>
  <sup>1</sup>SAIVT, Queensland University of Technology &nbsp;
  <sup>2</sup>Shield AI &nbsp;
  <sup>3</sup>La Trobe University
</p>

---

> [!NOTE]
> **Code release.** This repository currently contains the paper's poster, figures, and project description. The full codebase — training, evaluation, and pretrained checkpoints — will be uploaded **prior to CVPR**. Star or watch the repo for updates.

## TL;DR

> **One** GCD model. **Any** modality. **No** per-dataset fine-tuning.

OmniGCD is trained **once** on a synthetic clustering task and then performs Generalized Category Discovery zero-shot across **16 datasets spanning 4 modalities** (vision, text, audio, remote sensing) by pairing with off-the-shelf modality-specific encoders.

<p align="center">
  <img src="images/summary_diagram_v5.png" alt="OmniGCD summary diagram" width="90%"/>
</p>

## Abstract

Generalized Category Discovery (GCD) challenges methods to identify known and novel classes using partially labeled data, mirroring human category learning. Unlike prior GCD methods, which operate within a single modality and require dataset-specific fine-tuning, we propose a modality-agnostic GCD approach inspired by the human brain's abstract category formation. Our **OmniGCD** leverages modality-specific encoders (e.g., vision, audio, text, remote sensing) to process inputs, followed by dimension reduction to construct a **GCD latent space**, which is transformed at test-time into a representation better suited for clustering using a novel synthetically trained Transformer-based model. To evaluate OmniGCD, we introduce a **zero-shot GCD setting** where no dataset-specific fine-tuning is allowed, enabling modality-agnostic category discovery. **Trained once on synthetic data**, OmniGCD performs zero-shot GCD across 16 datasets spanning four modalities, improving classification accuracy for known and novel classes over baselines (average percentage point improvement of **+6.2**, **+17.9**, **+1.5** and **+12.7** for vision, text, audio and remote sensing). This highlights the importance of strong encoders while decoupling representation learning from category discovery. Improving modality-agnostic methods will propagate across modalities, enabling encoder development independent of GCD. Our work serves as a benchmark for future modality-agnostic GCD works, paving the way for scalable, human-inspired category discovery.

## Method

<p align="center">
  <img src="images/method_diagrma.png" alt="OmniGCD method diagram" width="90%"/>
</p>

OmniGCD treats GCD as a sequence-to-sequence clustering problem in a shared latent space:

1. **Encode.** Compute feature vectors of labelled and unlabelled samples using a frozen modality-specific encoder (DINOv1 / DINOv2 for vision, etc.) — these become **data tokens**.
2. **Map.** Project data tokens, via dimension reduction, into a common dimensionality.
3. **Tokenize labels.** Concatenate each data token with an embedded label token: a learned embedding for known classes, and a *masked* token for unknown samples.
4. **Form the GCD latent space.** Concatenate all tokens into one sequence — the *initial GCD latent space*.
5. **GCDformer.** A transformer optimizes the latent space so that same-class points cluster together — the *optimized GCD latent space*.
6. **Cluster.** k-means on the optimized space produces the final classifications.

GCDformer is trained **once** on a synthetic 2D clustering task and then applied — without any fine-tuning — to real datasets across all four modalities.

### Standard GCD vs. Zero-shot GCD

|                | Train-time access to $D_L$ | Train-time access to $D_U$ | Per-dataset fine-tuning |
|----------------|:-:|:-:|:-:|
| Standard GCD   | yes | yes | yes |
| Zero-shot GCD (ours) | no | no | **no** |

In zero-shot GCD, $D_L$ is only consulted at test time to provide the label tokens for known classes.

## Results

OmniGCD performs zero-shot GCD across **16 datasets** spanning **4 modalities**, compared against k-means and a no-fine-tuning GCD baseline.

<p align="center">
  <img src="images/results_table.png" alt="OmniGCD zero-shot GCD results across 16 datasets and 4 modalities" width="95%"/>
</p>

See the [poster](poster.pdf) and the [paper](https://arxiv.org/abs/2604.14762) for full details.

## Installation

> _Coming soon._ Setup instructions and a `requirements.txt` will be added with the code release.

```bash
git clone https://github.com/<user>/OmniGCD.git
cd OmniGCD
# pip install -r requirements.txt
```

## Training

> _Coming soon._ Single-command training of GCDformer on the synthetic GCD task.

```bash
# python train.py --config configs/omnigcd.yaml
```

## Evaluation (zero-shot)

> _Coming soon._ Evaluation of a trained OmniGCD checkpoint on any of the 16 supported datasets, with no fine-tuning.

```bash
# python eval.py --dataset cub --encoder dinov2_vitb14 --checkpoint <path>
```

## Poster

The CVPR poster is included in this repo: [`poster.pdf`](poster.pdf).

## Citation

If you find OmniGCD useful in your research, please cite:

```bibtex
@inproceedings{shipard2026omnigcd,
  title     = {OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism},
  author    = {Shipard, Jordan and Wiliem, Arnold and Nguyen Thanh, Kien and Xiang, Wei and Fookes, Clinton},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgements

This work was supported by the **SmartSat CRC** (funded by the Australian Government's CRC Program) and by **Shield AI**, a global leader in AI pilots for defence and civilian applications. We also build on prior work in Generalized Category Discovery, in particular Vaze et al. (CVPR 2022).
