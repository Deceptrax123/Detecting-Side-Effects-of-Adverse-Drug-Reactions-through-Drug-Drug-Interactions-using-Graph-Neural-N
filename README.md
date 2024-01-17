# Modelling Drug-Target Interaction to predict side effects using Graph Neural Networks

## About
We create graph neural networks of various types including self-supervised pretraining of models to predict side effects of a drug and target interaction, further we achieve state of the art results on various performance metrics.

## Dataset
TwoSides Dataset of TDC was employed for model training and testing. The code for the dataloader and batching policy can be found <a href="https://github.com/Deceptrax123/GNN-Dataloader-For-Chemical-Interaction-Applications">here</a>.

## Models
### Trained with randomly initialized weights
- Graph Convolution
- Graph Attention
- Chebyshev Convolutions
- SAGE Convolution
  
### Fine-tuned after SSL Pretraining using Variational Autoencoders
<a href="https://github.com/Deceptrax123/Graph-VAE">Link</a> to code and implementation of pre-training stages 
- Graph Convolution
- Chebyshev Convolution

### Ensemble of weights two pre-trained models

- Pretrained GCN Models  and Pretrained Chebyshev Convolution Models

## Contributors
Our Contributors to the research, model planning and creation, management and development
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Varenya007"><img src="https://avatars.githubusercontent.com/u/90688425?v=4?s=100" width="100px;" alt="Varenya"/><br /><sub><b>Varenya</b></sub></a><br /><a href="#data-Varenya007" title="Data">🔣</a> <a href="#research-Varenya007" title="Research">🔬</a> <a href="#code-Varenya007" title="Code">💻</a> <a href="#projectManagement-Varenya007" title="Project Management">📆</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Deceptrax123"><img src="https://avatars.githubusercontent.com/u/87447180?v=4?s=100" width="100px;" alt="Srinitish S"/><br /><sub><b>Srinitish S</b></sub></a><br /><a href="#data-Deceptrax123" title="Data">🔣</a> <a href="https://github.com/Deceptrax123/Drug-Interaction-Using-GNNs/commits?author=Deceptrax123" title="Code">💻</a> <a href="#research-Deceptrax123" title="Research">🔬</a> <a href="#maintenance-Deceptrax123" title="Maintenance">🚧</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->