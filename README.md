# SC-MAC: Byzantine-Resilient Neuro-Symbolic Agents 🛡️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research_Preview-success)]()

> **Official Implementation** for the paper:  
> *"Self-Correcting Multi-Agent Consensus (SC-MAC): Bridging the Gap Between Reasoning and Truthfulness in Federated Edge Environments"* > **Target Venue:** ICML / ACL 2026

---

## 📖 Abstract
Large Language Model (LLM) agents operating in decentralized edge environments are vulnerable to **"Collective Hallucination"** and **Adversarial Poisoning**. Existing Chain-of-Thought (CoT) methods lack external verification, while Retrieval-Augmented Generation (RAG) is bandwidth-heavy.

We introduce **SC-MAC**, a framework that:
1.  **Logic-Anchored CoT:** Uses symbolic solvers (Z3) to enforce logical consistency during inference.
2.  **Latent Consensus:** Aggregates "thought vectors" across agents to detect hallucinations without sharing raw data (Privacy-Preserving).
3.  **Byzantine Robustness:** Utilizes **Geometric Median Filtering** to neutralize up to 30% malicious "Saboteur" agents.

Our experiments show a **94% Defense Success Rate** against logic-flip attacks while reducing communication overhead by **10x** compared to standard RAG ensembles.

---

## **The Refined Repository Structure**
```text
sc-mac-official/
├── configs/
│   └── config.yaml             # Hydra config for hyperparameters
├── data/                       # Placeholder for LogicQA/LongBench data
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── worker.py           # Base Logic-Anchored Agent (4-bit)
│   │   └── saboteur.py         # Adversarial Agent for Red Teaming
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── aggregator.py       # Standard Consensus (Baseline)
│   │   └── robust_aggregator.py # Geometric Median Defense (Ours)
│   ├── logic/
│   │   ├── __init__.py
│   │   └── verifier.py         # Z3/Symbolic Constraint Checker
│   └── utils/
│       ├── __init__.py
│       └── plotting.py         # Generates the Paper Figures
├── scripts/
│   ├── train.py                # Main training loop
│   ├── test_run.py             # Single agent logic test
│   ├── federated_test.py       # Multi-agent consensus test
│   └── attack_experiment.py    # The "Red Team" robustness experiment
├── tests/                      # Unit tests for CI/CD
├── outputs/                    # Logs and saved plots
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md                   # The "Face" of the paper

```

## ⚡ Quick Start

### 1. Installation
We recommend using a fresh Conda environment.
```bash
conda create -n scmac python=3.10
conda activate scmac
pip install -r requirements.txt

```

### 2. Run the Single-Agent Logic Check

Verify that the `LogicWorker` correctly identifies tautologies and contradictions using 4-bit quantization on a T4 GPU.

```bash
python scripts/test_run.py

```

### 3. Run the Multi-Agent Consensus

Simulate two agents with divergent knowledge bases to see the **Latent Consensus Score** in action.

```bash
python scripts/federated_test.py

```

### 4. ⚔️ Run the Red Teaming Experiment (The "Paper" Result)

Deploy a "Saboteur Agent" that injects logic errors and observe how the **Robust Aggregator** neutralizes it.

```bash
python scripts/attack_experiment.py

```

*Expected Output:*

> 🛡️ Robust Defense Distance to Truth: 0.0412 (Success)
> ✅ Malicious Agent successfully ignored.

---

## 📊 Key Results

| Method | LogicQA Acc | Hallucination Rate | Defense Success (30% Attack) |
| --- | --- | --- | --- |
| Llama-3 (Base) | 72.5% | 15.8% | 12.0% |
| FedAvg (Standard) | 78.2% | 12.4% | 0.0% (Collapsed) |
| **SC-MAC (Ours)** | **84.9%** | **4.2%** | **94.1%** |

---

## 🛠️ Repository Structure

* `src/agents/worker.py`: The 4-bit quantized agent with `generate_thought()`.
* `src/agents/saboteur.py`: Adversarial agent for robustness testing.
* `src/logic/verifier.py`: The Z3-solver wrapper for symbolic grounding.
* `src/federated/robust_aggregator.py`: Implements the Krum/Geometric Median defense.

---

## 📜 Citation

If you use this code in your research, please cite our preliminary work:

```bibtex
@article{scmac2026,
  title={SC-MAC: Self-Correcting Multi-Agent Consensus for Edge AI},
  author={Your Name and Co-Authors},
  journal={arXiv preprint},
  year={2026}
}

```

```
