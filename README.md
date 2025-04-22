# AI4Polymer [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

A curated list of awesome papers, tools, authors, books, blogs and other resources related to AI4Polymer.

Inspired by [awesome-python](https://awesome-python.com) and [awesome-python-chemistry](https://github.com/lmmentel/awesome-python-chemistry).

## Table of contents

- [Awesome AI for polymer](#ai4polymer-)
  - [Large language model](#large-language-model)
  - [Polymer design](#polymer-design)
  - [Polymer dataset](#polymer-dataset)
  - [Polymer representation](#polymer-representation)
  - [Generative model](#generative-model)
  - [Multi-task learning](#multi-task-learning)
  - [Polymer properties prediction](#polymer-properties-prediction)
  - [AI4Polymer review](#ai4polymer-review)
  - [AI4Polymer books or chapter](#ai4polymer-books-or-chapter)
  - [Wechat](#个人微信公众号)

## Large language model

*Paper of using Large language model (LLM) for polymer.*

- [Unified Multimodal Multidomain Polymer Representation for Property Prediction](https://www.researchsquare.com/article/rs-5795287/v1) - A paper of using LLM with Unified Multimodal Multidomain for polymer property prediction.


## Polymer design
- [On-demand reverse design of polymers with PolyTAO](https://www.nature.com/articles/s41524-024-01466-5#citeas) [2024]

## Polymer dataset

- [Polyinfo](https://ieeexplore.ieee.org/abstract/document/6076416) - A Japanese website collecting diverse experimental data for polymers from academic papers. PoLyInfo collects information on polymer names, chemical structures, processing methods of samples, measurement conditions, properties, used monomers and polymerization methods.

## Polymer representation

### Fingerprints or descriptors
- ECFP ([Extended-Connectivity Fingerprints](https://pubs.acs.org/doi/10.1021/ci100050t)) - Extended-connectivity fingerprints (ECFPs) is a class of topological fingerprints for molecular characterization. Historically, topological fingerprints were developed for substructure and similarity searching. Their features represent the presence of particular substructures, allowing easier interpretation of analysis results.
- Mordred ([Mordred: a molecular descriptor calculator](https://doi.org/10.1186/s13321-018-0258-y)) [2018] - Mordred is an open-source tool capable of calculating constitutional descriptors (e.g., molecular weight and atom counts), topological descriptors (information about molecular connectivity and branching), geometrical descriptors (three-dimensional shape and size parameters), electronic descriptors (properties related to electron distribution within the molecule), and physicochemical descriptors (such as solubility parameters and polar surface area).

### PSMILES
- [Enhancing Copolymer Property Prediction through the Weighted-Chained-SMILES Machine Learning Framework](https://doi.org/10.1021/acsapm.3c02715) - The authors use weighted-chained-SMILES to represent copolymers.
- Polymer genome ([Machine-learning predictions of polymer properties with Polymer Genome](https://doi.org/10.1063/5.0023759)) [2020] - Heirarchical fingerprints.
### BigSMILES
### GNN
- [A graph representation of molecular ensembles for polymer property prediction](http://dx.doi.org/10.1039/D2SC02839E) - The authors expand molecular graph representations by incorporating “stochastic” edges to describe the average structure of the repeating unit. In effect, these stochastic edges are bonds weighted by their probability of occurring in the polymer chain. This representation can capture (i) the recurrent nature of polymers' repeating units, (ii) the different topologies and isomerisms of polymer chains, and (iii) their varying monomer composition and stoichiometry.
- [Representing Polymers as Periodic Graphs with Learned Descriptors for Accurate Polymer Property Predictions](https://doi.org/10.1021/acs.jcim.2c00875) - Representing the polymer as a circular graph by linking the head and the tail of the monomer repeating unit.
### 3D geometry
- [Uni-mol]

### LLM
- [TransPolymer](https://doi.org/10.1038/s41524-023-01016-5) - A large language model based on RoBERTa architecture for polymer representation and property prediction.
- [PolyBERT](https://doi.org/10.1038/s41467-023-39868-6) - A large language model based on DeBERTa architecture for polymer representation and multi-task property prediction.
### Multi-modality
- [Multimodal Transformer for Property Prediction in Polymers](https://doi.org/10.1021/acsami.4c01207) - PSMILES, 2D Graph
- [Mmpolymer: A multimodal multitask pretraining framework for polymer property prediction](https://arxiv.org/abs/2406.04727) - SMILES, 3D geometry
- [Multimodal machine learning with large language embedding model for polymer property prediction](https://arxiv.org/abs/2503.22962) [2025] - PSMILES, 3D geometry (Uni-Mol), LLM (Llama) encodering latent vectors
### Multi-domain
- Uni-Poly [Unified Multimodal Multidomain Polymer Representation for Property Prediction](https://www.researchsquare.com/article/rs-5795287/v1) - SMILES, 2D graphs, 3D geometries, Morgan fingerprints and polymer domain-specific textual descriptions.
### Review or benchmarking
- [Evaluating Polymer Representations via Quantifying Structure–Property Relationships](https://doi.org/10.1021/acs.jcim.9b00358)

### Others
- [Ring Repeating Unit: An Upgraded Structure Representation of Linear Condensation Polymers for Property Prediction](https://doi.org/10.1021/acs.jcim.2c01389) [2023]

## Generative model
- [PolyConf: Unlocking Polymer Conformation Generation through Hierarchical Generative Models](https://arxiv.org/pdf/2504.08859) [2025] - Masked generative model, SE(3)/SO(3) diffusion model

## Multi-task learning
- [Polymer Informatics at Scale with Multitask Graph Neural Networks](https://doi.org/10.1021/acs.chemmater.2c02991) [2023]
- [Polymer graph neural networks for multitask property learning](https://doi.org/10.1038/s41524-023-01034-3) [2023]

## Polymer properties prediction

### Benchmarking
- [Estimation and Prediction of the Polymers’ Physical Characteristics Using the Machine Learning Models](https://www.mdpi.com/2073-4360/16/1/115) [2024]

### Tg
- [Benchmarking Machine Learning Models for Polymer Informatics: An Example of Glass Transition Temperature](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01031?ref=PDF) [2021]
- [Machine-Learning-Based Predictive Modeling of Glass Transition Temperatures: A Case of Polyhydroxyalkanoate Homopolymers and Copolymers](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00807) [2019]

### Dielectric properties
- [Prediction of polymer properties using infinite chain descriptors (ICD) and machine learning: Toward optimized dielectric polymeric materials](https://onlinelibrary.wiley.com/doi/10.1002/polb.24117) [2016]

### Others
- [Refractive index prediction models for polymers using machine learning](https://pubs.aip.org/aip/jap/article/127/21/215105/566123/Refractive-index-prediction-models-for-polymers) [2020] - Refractive index

## AI4Polymer review
- [Machine Learning in Polymer Research](https://doi.org/10.1002/adma.202413695) [2025]
- [Machine Learning Approaches in Polymer Science: Progress and Fundamental for a New Paradigm](https://onlinelibrary.wiley.com/doi/10.1002/smm2.1320) [2025]
- [Convergence of Artificial Intelligence, Machine Learning, Cheminformatics, and Polymer Science in Macromolecules](https://doi.org/10.1021/acs.macromol.4c01704) [2024]
- [Emerging Trends in Machine Learning: A Polymer Perspective](https://pubs.acs.org/doi/10.1021/acspolymersau.2c00053) [2023]
- [The Rise of Machine Learning in Polymer Discovery](https://doi.org/10.1002/aisy.202200243) [2023]
- [Machine learning for polymeric materials: an introduction](https://doi.org/10.1002/pi.6345) [2022]
- [Polymer informatics: Current status and critical next steps](https://www.sciencedirect.com/science/article/abs/pii/S0927796X2030053X) [2021]
- [Machine learning in polymer informatics](https://doi.org/10.1002/inf2.12167) [2021]
- [Polymer Informatics: Opportunities and Challenges](https://doi.org/10.1021/acsmacrolett.7b00228) [2017]

## AI4Polymer books or chapter
- [AI Application Potential and Prospects in Materials Science: A Focus on Polymers](https://www.okipublishing.com/book/index.php/okip/catalog/book/62) [2025]

## 个人微信公众号

![鲸落生](wechat.jpg)


