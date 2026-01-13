# SC-MAC
Self-Correcting Multi-Agent Consensus (SC-MAC): Bridging the Gap Between Reasoning and Truthfulness in Federated Edge Environments


## 
```bash
sc-mac-official/
├── configs/                # Hydra/YAML configuration files
│   ├── experiment/         # Specific hyperparameter sets for ACL/ICML runs
│   └── model/              # Model architectures (Worker vs. Auditor)
├── data/                   # Data loaders and preprocessing scripts
├── src/                    # Core source code
│   ├── agents/             # Agent logic (CoT, Consensus mechanisms)
│   ├── federated/          # FedAvg, Peer-to-peer communication logic
│   ├── logic/              # Symbolic logic checkers & Latent verifiers
│   └── utils/              # Metrics (TruthfulQA-Score), logging, and plotting
├── notebooks/              # For visualization and qualitative analysis
├── tests/                  # Unit tests for the CACC loss and Logic-Guard
├── scripts/                # Entry points for training/eval (e.g., train.py)
├── requirements.txt
└── README.md               # The "Face" of the paper (Metrics & Figures)
```