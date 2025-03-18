# **Awesome ICLR 2025 Molecular ML Paper Collection**

This repo contains a comprehensive compilation of **molecular ML** papers that were accepted at the [The Thirteenth International Conference on Learning Representations](https://iclr.cc/). Molecular machine learning (Molecular ML) leverages AI to predict chemical properties and accelerate drug discovery, enabling faster, cost-effective advancements in healthcare and materials science. 

**Short Overview**: We've got ~150 papers focusing on molecular ML in ICLR'25. This year's ICLR emphasize the convergence of machine learning, molecular modeling, and biological sciences, showcasing innovations across generative models, optimization, and representation learning. Key focus areas include **protein language modeling**, **3D molecular generation**, **molecular property prediction**, and **graph-based approaches** for molecular dynamics and design. These works advance techniques such as diffusion models, geometric learning, and multi-modal transfer learning to address challenges in **drug discovery, RNA, Antibody Design, protein engineering**, and **single-cell genomics**, paving the way for faster, more accurate predictions and designs in molecular biology and chemistry.


**Have a look and throw me a review (and, a star ⭐, maybe!)** Thanks!


---



## **All Topics:** 

<details open>
  <summary><b>View Topic list!</b></summary>

- [Molecule Generation](#Generative)
- [Diffusion Models](#Diffusion)
- [Flow-Matching](#Flow-Matching)
- [Multi-modal Models](#Multi-modal)
- [Interactions](#Interactions)
- [Single-cell Application Works](#Single-cell)
- [Graphs and GNNs](#ggnns)
- [Protein Language Models](#PLMs)
- [Language Models in Molecular ML](#LMs)
- [Property Prediction and Optimization](#|Property)
- [DNA](#DNA)
- [RNA](#RNA)
- [Antibody](#Antibody)
- [Chemistry and Chemical Models](#chem)
- [3D Modeling and Representation Learning](#3D)
- [Others](#Others)
</details>



<a name="Generative" />

## Generative Modeling
- [IgGM: A Generative Model for Functional Antibody and Nanobody Design](https://openreview.net/pdf?id=zmmfsJpYcq)
- [Generative Flows on Synthetic Pathway for Drug Design](https://openreview.net/pdf?id=pB1XSj2y4X)
- [GANDALF: Generative AttentioN based Data Augmentation and predictive modeLing Framework for personalized cancer treatment](https://openreview.net/pdf?id=WwmtcGr4lP)
- [Proteina: Scaling Flow-based Protein Structure Generative Models](https://openreview.net/pdf?id=TVQLu34bdw)
- [Generator Matching: Generative modeling with arbitrary Markov processes](https://openreview.net/pdf?id=RuP17cJtZo)
- [Measuring And Improving Persuasiveness Of Generative Models](https://openreview.net/pdf?id=NfCEVihkdC)
- [Bio-xLSTM: Generative modeling, representation and in-context learning of biological and chemical sequences](https://openreview.net/pdf?id=IjbXZdugdj)
- [TFG-Flow: Training-free Guidance in Multimodal Generative Flow](https://openreview.net/pdf?id=GK5ni7tIHp)
- [Diffusion Generative Modeling for Spatially Resolved Gene Expression Inference from Histology Images](https://openreview.net/pdf?id=FtjLUHyZAO)
- [Trivialized Momentum Facilitates Diffusion Generative Modeling on Lie Groups](https://openreview.net/pdf?id=DTatjJTDl1)
- [Beyond Squared Error: Exploring Loss Design for Enhanced Training of Generative Flow Networks](https://openreview.net/pdf?id=4NTrco82W0)
- [Conformal Generative Modeling with Improved Sample Efficiency through Sequential Greedy Filtering](https://openreview.net/pdf?id=1i6lkavJ94)

<a name="Diffusion" />

### Diffusion Models
- [Discrete Diffusion Schrodinger Bridge Matching for Graph Transformation](https://openreview.net/pdf?id=tQyh0gnfqW)
- [NExT-Mol: 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation](https://openreview.net/pdf?id=p66a00KLWN)
- [Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs](https://openreview.net/pdf?id=mYgoNEsUDi)
- [Simple and Controllable Uniform Discrete Diffusion Language Models](https://openreview.net/pdf?id=i5MrJ6g5G1)
- [SymDiff: Equivariant Diffusion via Stochastic Symmetrisation](https://openreview.net/pdf?id=i1NNCrRxdM)
- [Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization](https://openreview.net/pdf?id=a6U41REOa5)
- [Unlocking Guidance for Discrete State-Space Diffusion and Flow Models](https://openreview.net/pdf?id=XsgHl54yO7)
- [Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers](https://openreview.net/pdf?id=WQV9kB1qSU)
- [Self-Supervised Diffusion Processes for Electron-Aware Molecular Representation Learning](https://openreview.net/pdf?id=UQ0RqfhgCk)
- [MorphoDiff: Cellular Morphology Painting with Diffusion Models](https://openreview.net/pdf?id=PstM8YfhvI)
- [Fast Direct: Query-Efficient  Online Black-box Guidance  for Diffusion-model Target Generation](https://openreview.net/pdf?id=OmpTdjl7RV)
- [Steering Masked Discrete Diffusion Models via Discrete Denoising Posterior Prediction](https://openreview.net/pdf?id=Ombm8S40zN)
- [ProtPainter: Draw or Drag Protein via Topology-guided Diffusion](https://openreview.net/pdf?id=Nq7yKYL0Bp)
- [Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design](https://openreview.net/pdf?id=G328D1xt4W)
- [Diffusion Generative Modeling for Spatially Resolved Gene Expression Inference from Histology Images](https://openreview.net/pdf?id=FtjLUHyZAO)
- [Trivialized Momentum Facilitates Diffusion Generative Modeling on Lie Groups](https://openreview.net/pdf?id=DTatjJTDl1)
- [DPLM-2: A Multimodal Diffusion Protein Language Model](https://openreview.net/pdf?id=5z9GjHgerY)
- [Chemistry-Inspired Diffusion with Non-Differentiable Guidance](https://openreview.net/pdf?id=4dAgG8ma3B)
- [The Superposition of Diffusion Models](https://openreview.net/pdf?id=2o58Mbqkd2)



## Flow-Matching
- [AssembleFlow: Rigid Flow Matching with Inertial Frames for Molecular Assembly](https://openreview.net/pdf?id=jckKNzYYA6)
- [MOFFlow: Flow Matching for Structure Prediction of Metal-Organic Frameworks](https://openreview.net/pdf?id=dNT3abOsLo)
- [Meta Flow Matching: Integrating Vector Fields on the Wasserstein Manifold](https://openreview.net/pdf?id=9SYczU3Qgm)
- [Stiefel Flow Matching for Moment-Constrained Structure Elucidation](https://openreview.net/pdf?id=84WmbzikPP)



<a name="Multi-modal" />

## Multi-modal Models
- [MolSpectra: Pre-training 3D Molecular Representation with Multi-modal Energy Spectra](https://openreview.net/pdf?id=xJDxVDG3x2)
- [Generating Multi-Modal and Multi-Attribute Single-Cell Counts with CFGen](https://openreview.net/pdf?id=3MnMGLctKb)



## Interactions
- [Boltzmann-Aligned Inverse Folding Model as a Predictor of Mutational Effects on Protein-Protein Interactions](https://openreview.net/pdf?id=lzdFImKK8w)
- [Exact Computation of Any-Order Shapley Interactions for Graph Neural Networks](https://openreview.net/pdf?id=9tKC0YM8sX)


## Single-cell
- [Generating Multi-Modal and Multi-Attribute Single-Cell Counts with CFGen](https://openreview.net/pdf?id=3MnMGLctKb)



<a name="ggnns" />

## Graphs and GNNs
- [MAGE: Model-Level Graph Neural Networks Explanations via Motif-based Graph Generation](https://openreview.net/pdf?id=vue9P1Ypk6)
- [Lift Your Molecules: Molecular Graph Generation in Latent Euclidean Space](https://openreview.net/pdf?id=uNomADvF3s)
- [Discrete Diffusion Schrodinger Bridge Matching for Graph Transformation](https://openreview.net/pdf?id=tQyh0gnfqW)
- [Graph Transformers Dream of Electric Flow](https://openreview.net/pdf?id=rWQDzq3O5c)
- [Homomorphism Counts as Structural Encodings for Graph Learning](https://openreview.net/pdf?id=qFw2RFJS5g)
- [Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges](https://openreview.net/pdf?id=pXN8T5RwNN)
- [Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs](https://openreview.net/pdf?id=mYgoNEsUDi)
- [CBGBench: Fill in the Blank of Protein-Molecule Complex Binding Graph](https://openreview.net/pdf?id=mOpNrrV2zH)
- [Towards Synergistic Path-based Explanations for Knowledge Graph Completion: Exploration and Evaluation](https://openreview.net/pdf?id=WQvkqarwXi)
- [REBIND: Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring](https://openreview.net/pdf?id=WNIEr5kydF)
- [A Theoretically-Principled Sparse, Connected, and Rigid Graph Representation of Molecules](https://openreview.net/pdf?id=OIvg3MqWX2)
- [From GNNs to Trees: Multi-Granular Interpretability for Graph Neural Networks](https://openreview.net/pdf?id=KEUPk0wXXe)
- [Exact Computation of Any-Order Shapley Interactions for Graph Neural Networks](https://openreview.net/pdf?id=9tKC0YM8sX)
- [GRAIN: Exact Graph Reconstruction from Gradients](https://openreview.net/pdf?id=7bAjVh3CG3)
- [Graph Sparsification via Mixture of Graphs](https://openreview.net/pdf?id=7ANDviElAo)
- [GotenNet: Rethinking Efficient 3D Equivariant Graph Neural Networks](https://openreview.net/pdf?id=5wxCQDtbMo)
- [Charting the Design Space of Neural Graph Representations for Subgraph Matching](https://openreview.net/pdf?id=5pd78GmXC6)
- [Pushing the Limits of All-Atom Geometric Graph Neural Networks: Pre-Training, Scaling, and Zero-Shot Transfer](https://openreview.net/pdf?id=4S2L519nIX)
- [Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Information Bottleneck](https://openreview.net/pdf?id=3kiZ5S5WkY)
- [Beyond Graphs: Can Large Language Models Comprehend Hypergraphs?](https://openreview.net/pdf?id=28qOQwjuma)
- [PharmacoMatch: Efficient 3D Pharmacophore Screening via Neural Subgraph Matching](https://openreview.net/pdf?id=27Qk18IZum)
- [InversionGNN: A Dual Path Network for Multi-Property Molecular Optimization](https://openreview.net/pdf?id=nYPuSzGE3X)


<a name="LMs" />

## Language Models in Molecular ML
- [LICO: Large Language Models for In-Context Molecular Optimization](https://openreview.net/pdf?id=yu1vqQqKkx)
- [Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo](https://openreview.net/pdf?id=xoXn62FzD0)
- [Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning](https://openreview.net/pdf?id=rQ7fz9NO7f)
- [NExT-Mol: 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation](https://openreview.net/pdf?id=p66a00KLWN)
- [Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language Models](https://openreview.net/pdf?id=mMhZS7qt0U)
- [ChemAgent: Self-updating Memories in Large Language Models Improves Chemical Reasoning](https://openreview.net/pdf?id=kuhIqeVg0e)
- [The OMG dataset: An Open MetaGenomic corpus for mixed-modality genomic language modeling](https://openreview.net/pdf?id=jlzNb1iWs3)
- [Simple and Controllable Uniform Discrete Diffusion Language Models](https://openreview.net/pdf?id=i5MrJ6g5G1)
- [Eliminating Position Bias of Language Models: A Mechanistic Approach](https://openreview.net/pdf?id=fvkElsJOsN)
- [Efficient Evolutionary Search Over Chemical Space with Large Language Models](https://openreview.net/pdf?id=awWiNvQwf3)
- [Concept Bottleneck Language Models For Protein Design](https://openreview.net/pdf?id=Yt9CFhOOFe)
- [Protein Language Model Fitness is a Matter of Preference](https://openreview.net/pdf?id=UvPdpa4LuV)
- [Metalic: Meta-Learning In-Context with Protein Language Models](https://openreview.net/pdf?id=TUKt7ag0qq)
- [Structure Language Models for Protein Conformation Generation](https://openreview.net/pdf?id=OzUNDnpQyd)
- [HELM: Hierarchical Encoding for mRNA Language Modeling](https://openreview.net/pdf?id=MMHqnUOnl0)
- [SMI-Editor: Edit-based SMILES Language Model with Fragment-level Supervision](https://openreview.net/pdf?id=M29nUGozPa)
- [RetroInText: A Multimodal Large Language Model Enhanced Framework for Retrosynthetic Planning via In-Context Representation Learning](https://openreview.net/pdf?id=J6e4hurEKd)
- [OSDA Agent: Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agents](https://openreview.net/pdf?id=9YNyiCJE3k)
- [NutriBench: A Dataset for Evaluating Large Language Models in Nutrition Estimation from Meal Descriptions](https://openreview.net/pdf?id=6LtdZCyuZR)
- [DPLM-2: A Multimodal Diffusion Protein Language Model](https://openreview.net/pdf?id=5z9GjHgerY)
- [Beyond Graphs: Can Large Language Models Comprehend Hypergraphs?](https://openreview.net/pdf?id=28qOQwjuma)

<a name="PLMs" />

## Protein Language Models
- [Protein Language Model Fitness is a Matter of Preference](https://openreview.net/pdf?id=UvPdpa4LuV)
- [Metalic: Meta-Learning In-Context with Protein Language Models](https://openreview.net/pdf?id=TUKt7ag0qq)
- [DPLM-2: A Multimodal Diffusion Protein Language Model](https://openreview.net/pdf?id=5z9GjHgerY)



<a name="Property" />

## Property Prediction and Optimization
- [InversionGNN: A Dual Path Network for Multi-Property Molecular Optimization](https://openreview.net/pdf?id=nYPuSzGE3X)
- [CL-MFAP: A Contrastive Learning-Based Multimodal Foundation Model for Molecular Property Prediction and Antibiotic Screening](https://openreview.net/pdf?id=fv9XU7CyN2)
- [UniGEM: A Unified Approach to Generation and Property Prediction for Molecules](https://openreview.net/pdf?id=Lb91pXwZMR)
- [Beyond Sequence: Impact of Geometric Context for RNA Property Prediction](https://openreview.net/pdf?id=9htTvHkUhh)
- [Curriculum-aware Training for Discriminating Molecular Property Prediction Models](https://openreview.net/pdf?id=6DHIkLv5i3)
- [LICO: Large Language Models for In-Context Molecular Optimization](https://openreview.net/pdf?id=yu1vqQqKkx)
- [Optimistic Games for Combinatorial Bayesian Optimization with Application to Protein Design](https://openreview.net/pdf?id=xiyzCfXTS6)
- [Data Distillation for extrapolative protein design through exact preference optimization](https://openreview.net/pdf?id=ua5MHdsbck)
- [SOO-Bench: Benchmarks for Evaluating the Stability of Offline Black-Box Optimization](https://openreview.net/pdf?id=bqf0aCF3Dd)
- [Searching for Optimal Solutions with LLMs via Bayesian Optimization](https://openreview.net/pdf?id=aVfDrl7xDV)
- [Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization](https://openreview.net/pdf?id=a6U41REOa5)
- [Latent Bayesian Optimization via Autoregressive Normalizing Flows](https://openreview.net/pdf?id=ZCOwwRAaEl)
- [Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design](https://openreview.net/pdf?id=G328D1xt4W)
- [Multi-objective antibody design with constrained preference optimization](https://openreview.net/pdf?id=4ktJJBvvUd)



## DNA
- [Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design](https://openreview.net/pdf?id=G328D1xt4W)


## RNA
- [gRNAde: Geometric Deep Learning for 3D RNA inverse design](https://openreview.net/pdf?id=lvw3UgeVxS)
- [Size-Generalizable RNA Structure Evaluation by Exploring Hierarchical Geometries](https://openreview.net/pdf?id=QaTBHSqmH9)
- [HELM: Hierarchical Encoding for mRNA Language Modeling](https://openreview.net/pdf?id=MMHqnUOnl0)
- [KinPFN: Bayesian Approximation of RNA Folding Kinetics using Prior-Data Fitted Networks](https://openreview.net/pdf?id=E1m5yGMOiV)
- [DEPfold: RNA Secondary Structure Prediction as Dependency Parsing.](https://openreview.net/pdf?id=DpLFmc09pC)
- [Beyond Sequence: Impact of Geometric Context for RNA Property Prediction](https://openreview.net/pdf?id=9htTvHkUhh)

## Antibody
- [IgGM: A Generative Model for Functional Antibody and Nanobody Design](https://openreview.net/pdf?id=zmmfsJpYcq)
- [Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization](https://openreview.net/pdf?id=a6U41REOa5)
- [A Simple yet Effective $\\Delta\\Delta G$ Predictor is An Unsupervised Antibody Optimizer and Explainer](https://openreview.net/pdf?id=IxmWIkcKs5)
- [Multi-objective antibody design with constrained preference optimization](https://openreview.net/pdf?id=4ktJJBvvUd)

<a name="3D" />

## 3D Modeling and Representation Learning
- [DenoiseVAE: Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-training](https://openreview.net/pdf?id=ym7pr83XQr)
- [MolSpectra: Pre-training 3D Molecular Representation with Multi-modal Energy Spectra](https://openreview.net/pdf?id=xJDxVDG3x2)
- [When Selection meets Intervention: Additional Complexities in Causal Discovery](https://openreview.net/pdf?id=xByvdb3DCm)
- [NExT-Mol: 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation](https://openreview.net/pdf?id=p66a00KLWN)
- [gRNAde: Geometric Deep Learning for 3D RNA inverse design](https://openreview.net/pdf?id=lvw3UgeVxS)
- [SOO-Bench: Benchmarks for Evaluating the Stability of Offline Black-Box Optimization](https://openreview.net/pdf?id=bqf0aCF3Dd)
- [Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport](https://openreview.net/pdf?id=VGURexnlUL)
- [3DMolFormer: A Dual-channel Framework for Structure-based Drug Discovery](https://openreview.net/pdf?id=RgE1qiO2ek)
- [Not-So-Optimal Transport Flows for 3D Point Cloud Generation](https://openreview.net/pdf?id=62Ff8LDAJZ)
- [GotenNet: Rethinking Efficient 3D Equivariant Graph Neural Networks](https://openreview.net/pdf?id=5wxCQDtbMo)
- [PharmacoMatch: Efficient 3D Pharmacophore Screening via Neural Subgraph Matching](https://openreview.net/pdf?id=27Qk18IZum)
- [ProtComposer: Compositional Protein Structure Generation with 3D Ellipsoids](https://openreview.net/pdf?id=0ctvBgKFgc)






<a name="chem" />

## Chemistry and Chemical Models

- [A new framework for evaluating model out-of-distribution generalisation for the biochemical domain](https://openreview.net/pdf?id=qFZnAC4GHR)
- [ChemAgent: Self-updating Memories in Large Language Models Improves Chemical Reasoning](https://openreview.net/pdf?id=kuhIqeVg0e)
- [Efficient Evolutionary Search Over Chemical Space with Large Language Models](https://openreview.net/pdf?id=awWiNvQwf3)
- [Bio-xLSTM: Generative modeling, representation and in-context learning of biological and chemical sequences](https://openreview.net/pdf?id=IjbXZdugdj)
- [Chemistry-Inspired Diffusion with Non-Differentiable Guidance](https://openreview.net/pdf?id=4dAgG8ma3B)

## Others
- [CFD: Learning Generalized Molecular Representation via Concept-Enhanced  Feedback Disentanglement](https://openreview.net/pdf?id=CsOIYMOZaV)
- [CheapNet: Cross-attention on Hierarchical representations for Efficient protein-ligand binding Affinity Prediction](https://openreview.net/pdf?id=A1HhtITVEi)
- [Group Ligands Docking to Protein Pockets](https://openreview.net/pdf?id=zDC3iCBxJb)
- [MADGEN - Mass-Spec attends to De Novo Molecular generation](https://openreview.net/pdf?id=78tc3EiUrN)
- [Procedural Synthesis of Synthesizable Molecules](https://openreview.net/pdf?id=OGfyzExd69)
- [ProteinBench: A Holistic Evaluation of Protein Foundation Models](https://openreview.net/pdf?id=BksqWM8737)
- [Distilling Structural Representations into Protein Sequence Models](https://openreview.net/pdf?id=KXrgDM3mVD)
- [ShEPhERD: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design](https://openreview.net/pdf?id=KSLkFYHlYg)
- [Enhancing the Scalability and Applicability of Kohn-Sham Hamiltonians for Molecular Systems](https://openreview.net/pdf?id=twEvvkQqPS)
- [EVA: Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow](https://openreview.net/pdf?id=KHkBpvmYVI)
- [Equivariant Masked Position Prediction for Efficient Molecular Representation](https://openreview.net/pdf?id=Nue5iMj8n6)
- [Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs](https://openreview.net/pdf?id=RyWypcIMiE)
- [SynFlowNet: Design of Diverse and Novel Molecules with Synthesis Constraints](https://openreview.net/pdf?id=uvHmnahyp1)
- [Integrating Protein Dynamics into Structure-Based Drug Design via Full-Atom Stochastic Flows](https://openreview.net/pdf?id=9qS3HzSDNv)
- [Learning Molecular Representation in a Cell](https://openreview.net/pdf?id=BbZy8nI1si)
- [Deep Signature: Characterization of Large-Scale Molecular Dynamics](https://openreview.net/pdf?id=xayT1nn8Mg)
- [UniMatch: Universal Matching from Atom to Task for Few-Shot Drug Discovery](https://openreview.net/pdf?id=v9EjwMM55Y)
- [Multi-domain Distribution Learning for De Novo Drug Design](https://openreview.net/pdf?id=g3VCIM94ke)
- [Hyperbolic Genome Embeddings](https://openreview.net/pdf?id=NkGDNM8LB0)
- [E(3)-equivariant models cannot learn chirality: Field-based molecular generation](https://openreview.net/pdf?id=mXHTifc1Fn)
- [MAGNet: Motif-Agnostic Generation of Molecules from Scaffolds](https://openreview.net/pdf?id=5FXKgOxmb2)
- [Atomas: Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Generation](https://openreview.net/pdf?id=mun3bGqdDM)
- [Rethinking the generalization of drug target affinity prediction algorithms via similarity aware evaluation](https://openreview.net/pdf?id=j7cyANIAxV)
- [AtomSurf: Surface Representation for Learning on Protein Structures](https://openreview.net/pdf?id=ARQIJXFcTH)
- [Steering Protein Family Design through Profile Bayesian Flow](https://openreview.net/pdf?id=PSiijdQjNU)
- [Fast Uncovering of Protein Sequence Diversity from Structure](https://openreview.net/pdf?id=1iuaxjssVp)
- [Leveraging Discrete Structural Information for Molecule-Text Modeling](https://openreview.net/pdf?id=eGqQyTAbXC)
- [Learning to engineer protein flexibility](https://openreview.net/pdf?id=L238BAx0wP)

### Others (Can be outside scope)
- [An Information Criterion for Controlled Disentanglement of Multimodal Data](https://openreview.net/pdf?id=3n4RY25UWP)
- [Differentially private learners for heterogeneous treatment effects](https://openreview.net/pdf?id=1z3SOCwst9)
- [CURIE: Evaluating LLMs on Multitask Scientific Long-Context Understanding and Reasoning](https://openreview.net/pdf?id=jw2fC6REUB)
- [Boltzmann priors for Implicit Transfer Operators](https://openreview.net/pdf?id=pRCOZllZdT)
- [Contextualizing biological perturbation experiments through language](https://openreview.net/pdf?id=5WEpbilssv)
- [Rethinking the role of frames for SE(3)-invariant crystal structure modeling](https://openreview.net/pdf?id=gzxDjnvBDa)
- [Fast and Accurate Blind Flexible Docking](https://openreview.net/pdf?id=iezDdA9oeB)
- [Vector-ICL: In-context Learning with Continuous Vector Representations](https://openreview.net/pdf?id=xing7dDGh3)
- [STAR: Synthesis of Tailored Architectures](https://openreview.net/pdf?id=HsHxSN23rM)
- [Scalable Universal T-Cell Receptor Embeddings from Adaptive Immune Repertoires](https://openreview.net/pdf?id=wyF5vNIsO7)
- [Constructing Confidence Intervals for Average Treatment Effects from Multiple Datasets](https://openreview.net/pdf?id=BHFs80Jf5V)
- [TRENDy: Temporal Regression of Effective Nonlinear Dynamics](https://openreview.net/pdf?id=NvDRvtrGLo)
- [SAGEPhos: Sage Bio-Coupled and Augmented Fusion for Phosphorylation Site Detection](https://openreview.net/pdf?id=hLwcNSFhC2)
- [Redefining the task of Bioactivity Prediction](https://openreview.net/pdf?id=S8gbnkCgxZ)
- [Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning](https://openreview.net/pdf?id=vFanHFE4Qv)
- [CryoFM: A Flow-based Foundation Model for Cryo-EM Densities](https://openreview.net/pdf?id=T4sMzjy7fO)
- [Training Free Guided Flow-Matching with Optimal Control](https://openreview.net/pdf?id=61ss5RA1MM)
- [Bridging the Gap between Database Search and \\emph{De Novo} Peptide Sequencing with SearchNovo](https://openreview.net/pdf?id=SjMtxqdQ73)
- [In vivo cell-type and brain region classification via multimodal contrastive learning](https://openreview.net/pdf?id=10JOlFIPjt)
- [Nonlinear Sequence Embedding by Monotone Variational Inequality](https://openreview.net/pdf?id=U834XHJuqk)
- [Multi-Label Node Classification with Label Influence Propagation](https://openreview.net/pdf?id=3X3LuwzZrl)
- [Reinforcement Learning for Control of Non-Markovian Cellular Population Dynamics](https://openreview.net/pdf?id=dsHpulHpOK)
- [ReNovo: Retrieval-Based \\emph{De Novo} Mass Spectrometry Peptide Sequencing](https://openreview.net/pdf?id=uQnvYP7yX9)
- [When do GFlowNets learn the right distribution?](https://openreview.net/pdf?id=9GsgCUJtic)
- [cryoSPHERE: Single-Particle HEterogeneous REconstruction from cryo EM](https://openreview.net/pdf?id=n8O0trhost)
- [E(n) Equivariant Topological Neural Networks](https://openreview.net/pdf?id=Ax3uliEBVR)
- [MuHBoost: A Multi-Label Boosting Method For Practical Longitudinal Human Behavior Modeling](https://openreview.net/pdf?id=BAelAyADqn)
- [Fast unsupervised ground metric learning with tree-Wasserstein distance](https://openreview.net/pdf?id=FBhKUXK7od)
- [Interpretable Causal Representation Learning for Biological Data in the Pathway Space](https://openreview.net/pdf?id=3Fgylj4uqL)
- [No Equations Needed: Learning System Dynamics Without Relying on Closed-Form ODEs](https://openreview.net/pdf?id=kbm6tsICar)
- [Composing Unbalanced Flows for Flexible Docking and Relaxation](https://openreview.net/pdf?id=gHLWTzKiZV)
- [Tree-Wasserstein Distance for High Dimensional Data with a Latent Feature Hierarchy](https://openreview.net/pdf?id=nYjAzwor9R)
- [Learning Equivariant Non-Local Electron Density Functionals](https://openreview.net/pdf?id=FhBT596F1X)
- [Immunogenicity Prediction with Dual Attention Enables Vaccine Target Selection](https://openreview.net/pdf?id=hWmwL9gizZ)
- [Topological Blindspots: Understanding and Extending Topological Deep Learning Through the Lens of Expressivity](https://openreview.net/pdf?id=EzjsoomYEb)
- [Efficient Biological Data Acquisition through Inference Set Design](https://openreview.net/pdf?id=gVkX9QMBO3)
- [GlycanML: A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning](https://openreview.net/pdf?id=owEQ0FTfVj)
- [Hotspot-Driven Peptide Design via Multi-Fragment Autoregressive Extension](https://openreview.net/pdf?id=jqmptcSNVG)
- [Recovering Manifold Structure Using Ollivier Ricci Curvature](https://openreview.net/pdf?id=aX7X9z3vQS)
- [Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport](https://openreview.net/pdf?id=gQlxd3Mtru)
- [Towards Domain Adaptive Neural Contextual Bandits](https://openreview.net/pdf?id=LNkMWCEssX)
- [MeToken: Uniform Micro-environment Token Boosts Post-Translational Modification Prediction](https://openreview.net/pdf?id=noUF58SMra)


---


**Missing any paper?**
If any paper is absent from the list, please feel free to [mail](mailto:azminetoushik.wasi@gmail.com) or [open an issue](https://github.com/azminewasi/Awesome-MoML-NeurIPS24/issues/new/choose) or submit a pull request. I'll gladly add that! Also, If I mis-categorized, please knock!

---

## More Collectons:
- [**Awesome ICLR 2025 Graph Paper Collection**](https://github.com/azminewasi/Awesome-Graph-Research-ICLR2025)
- [**Awesome NeurIPS 2024 Graph Paper Collection**](https://github.com/azminewasi/Awesome-Graph-Research-NeurIPS2024)
- [**Awesome ICML 2024 Graph Paper Collection**](https://github.com/azminewasi/Awesome-Graph-Research-ICML2024)
- [**Awesome ICLR 2024 Graph Paper Collection**](https://github.com/azminewasi/Awesome-Graph-Research-ICLR2024)
- [**Awesome-LLMs-ICLR-24**](https://github.com/azminewasi/Awesome-LLMs-ICLR-24/)

---

## ✨ **Credits**
**Azmine Toushik Wasi**

 [![website](https://img.shields.io/badge/-Website-blue?style=flat-square&logo=rss&color=1f1f15)](https://azminewasi.github.io) 
 [![linkedin](https://img.shields.io/badge/LinkedIn-%320beff?style=flat-square&logo=linkedin&color=1f1f18)](https://www.linkedin.com/in/azmine-toushik-wasi/) 
 [![kaggle](https://img.shields.io/badge/Kaggle-%2320beff?style=flat-square&logo=kaggle&color=1f1f1f)](https://www.kaggle.com/azminetoushikwasi) 
 [![google-scholar](https://img.shields.io/badge/Google%20Scholar-%2320beff?style=flat-square&logo=google-scholar&color=1f1f18)](https://scholar.google.com/citations?user=X3gRvogAAAAJ&hl=en) 
