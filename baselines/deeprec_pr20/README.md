# Baseline: deeprec-pr20 (Paixão et al., Pattern Recognition 2020)

PyTorch port of the released SqueezeNet weights from github.com/thiagopx/deeprec-pr20,
evaluated with this repo's data loading, TSP solver and metrics so the numbers are
directly comparable to SeamNet. Separate uv environment; see `pr20.py` for what is
reproduced and the one deviation (LKH instead of Concorde).

The released weights are pickled `.npy` dicts; `safe_npy.py` loads them with a
restricted unpickler (NumPy arrays and containers only), never with `allow_pickle=True`.

Leakage rules: the CDIP-trained model is evaluated on D1/D2 only, the ISRI-trained model
on D1/D3 only (each dataset's source documents are excluded from its model's training set).

    uv sync
    uv run python pr20.py --weights <traindata>/cdip_0.2_1000_32x32/model/9.npy \
        --data ../../datasets/D2 --mode single --out results/cdip_D2_single.json
