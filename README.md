# Side Effect Detection using Graph Neural Networks and Self Supervised Learning
Official code of our paper <a href="https://ieeexplore.ieee.org/document/10543001">"Detecting Side Effects of Adverse Drug Reactions Through Drug-Drug Interactions Using Graph Neural Networks and Self-Supervised Learning"</a> published in IEEE Access.

## Abstract
Adverse Drug Reactions(ADRs) due to drug-drug interactions present a public health problem worldwide that deserves attention due to its impact on mortality, morbidity and healthcare costs. They have been a major challenge in healthcare with the ever increasing complexity of therapeutics and an aging population in many regions. At present, no standard method to detect such adverse drug reactions exist until otherwise reported by patients after the drug is released to the market. Further, several studies show that it is extremely challenging to detect these rare cases during clinical trials held before the drug is released. Therefore, a reliable and efficient technique to predict such side effects before the release of the drug to the market is the need of the hour. Through the power of Graph Neural Networks and the knowledge representation abilities of self supervised learning, we designed an effective framework to model drug-drug interactions by leveraging the spatial and physical properties of drugs by representing them as molecular graphs. Through this approach, we developed a technique that resembles the dynamics of a chemical interaction. On training and testing this approach on the TwoSIDES Polypharmacy Dataset by Theraputic Data Commons(TDC), we achieve state of the art results by obtaining a precision of 75% and accuracy of 90% on the test dataset. Further, we also perform a case study on the DrugBank dataset and compare our results on the interaction type prediction task in-order to validate our approach on the drug-drug interaction domain and achieve excellent results with a precision, F1 and accuracy of 99%. Our study and experimental approaches lays the groundwork for further research on side-effect prediction through drug-drug interaction and the use of Graph Neural Networks in the field of Molecular Biology.

## Dataset
TwoSides Polypharmacy Side Effects Dataset of TDC was employed for model training and testing. The code we designed for the dataloader and batching policy can be found <a href="https://github.com/Deceptrax123/GNN-Dataloader-For-Chemical-Interaction-Applications">here</a>.

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

### Ensemble of weights of two pre-trained models

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
