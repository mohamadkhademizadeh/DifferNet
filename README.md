# DifferNet — Normalizing-Flow Anomaly Detection

Research-style implementation of **DifferNet**-style anomaly detection using **Normalizing Flows** (RealNVP) over CNN features.
Train only on **normal** images; low-likelihood samples at test time are flagged as **anomalies**. Includes a **Streamlit app**.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Train on normal images
python scripts/train.py --train_dir data/normals --epochs 30 --out models/differnet.pt

# 2) Evaluate on a folder (normal+defect mixed)
python scripts/test.py --data_dir data/test --model models/differnet.pt --out out

# 3) Run the demo
streamlit run app/streamlit_app.py
```

### Data layout
```
data/
├── normals/               # training images (normal only)
└── test/                  # evaluation images (mixed normal/defect)
```

---

## Repo Layout
```
DifferNet/
├── app/streamlit_app.py
├── configs/default.yaml
├── models/
├── scripts/
│   ├── train.py
│   ├── test.py
│   └── visualize.py
├── utils/
│   ├── flow_blocks.py
│   ├── dataset.py
│   ├── train_utils.py
│   └── viz.py
├── tests/test_flow.py
├── requirements.txt
└── README.md
```

---

## Notes
- Backbone: **ResNet18** feature map (layer3) → flatten → flow input.
- Flow: stack of **RealNVP** coupling blocks with actnorm + permutations.
- Scoring: **negative log-likelihood** (NLL) as anomaly score; optional pixel heatmap via input gradient.
- This implementation is educational (clean + readable) rather than SOTA-tuned.
