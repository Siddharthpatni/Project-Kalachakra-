#!/bin/bash
# ==========================================
# Project Kalachakra — Research Paper Downloader
# ==========================================
# Run this script in your terminal:
#   chmod +x download_papers.sh && ./download_papers.sh
# ==========================================

BASE_DIR="$(cd "$(dirname "$0")" && pwd)/papers"

echo "=========================================="
echo " Project Kalachakra — Paper Downloader"
echo "=========================================="
echo ""

download() {
    local dir="$1"
    local name="$2"
    local url="$3"
    mkdir -p "$BASE_DIR/$dir"
    echo "  ↓ $name"
    curl -sL -o "$BASE_DIR/$dir/$name" "$url"
}

# --- Transformer Architectures ---
echo "[1/7] Transformer Architectures"
download "transformers" "attention_is_all_you_need.pdf" "https://arxiv.org/pdf/1706.03762"
download "transformers" "temporal_fusion_transformer.pdf" "https://arxiv.org/pdf/1912.09363"
download "transformers" "patchtst.pdf" "https://arxiv.org/pdf/2211.14730"
download "transformers" "informer.pdf" "https://arxiv.org/pdf/2012.07436"

# --- Graph Neural Networks & Knowledge Graphs ---
echo "[2/7] Graph Neural Networks & Knowledge Graphs"
download "graphs" "kg_survey_representation.pdf" "https://arxiv.org/pdf/2002.00388"
download "graphs" "gnn_kg_completion.pdf" "https://arxiv.org/pdf/2007.12374"
download "graphs" "deep_graph_representation.pdf" "https://arxiv.org/pdf/2304.05055"
download "graphs" "graphgps.pdf" "https://arxiv.org/pdf/2205.12454"
download "graphs" "tokengt.pdf" "https://arxiv.org/pdf/2207.02505"

# --- Bayesian Neural Networks & Uncertainty ---
echo "[3/7] Bayesian Neural Networks & Uncertainty"
download "bayesian" "bayes_by_backprop.pdf" "https://arxiv.org/pdf/1505.05424"
download "bayesian" "uncertainty_quantification_dl.pdf" "https://arxiv.org/pdf/2405.20550"

# --- Neural ODE ---
echo "[4/7] Neural Ordinary Differential Equations"
download "neural_ode" "neural_ode.pdf" "https://arxiv.org/pdf/1806.07366"

# --- Quantum Machine Learning ---
echo "[5/7] Quantum Machine Learning"
download "quantum" "qaoa.pdf" "https://arxiv.org/pdf/1411.4028"
download "quantum" "supervised_qml_kernels.pdf" "https://arxiv.org/pdf/2101.11020"
download "quantum" "qml_survey.pdf" "https://arxiv.org/pdf/2310.10315"
download "quantum" "qaoa_supremacy.pdf" "https://arxiv.org/pdf/1602.07674"

# --- Causal Inference ---
echo "[6/7] Causal Inference & Discovery"
download "causal" "causal_ml_survey.pdf" "https://arxiv.org/pdf/2206.15475"
download "causal" "causal_discovery_survey.pdf" "https://arxiv.org/pdf/2305.10032"

# --- Explainability & Diffusion ---
echo "[7/7] Explainability & Diffusion Models"
download "xai" "shap.pdf" "https://arxiv.org/pdf/1705.07874"
download "diffusion" "ddpm.pdf" "https://arxiv.org/pdf/2006.11239"
download "diffusion" "score_sde.pdf" "https://arxiv.org/pdf/2011.13456"

echo ""
echo "=========================================="
echo " Download Complete!"
echo "=========================================="
total=$(find "$BASE_DIR" -name "*.pdf" -type f | wc -l | tr -d ' ')
echo "  Total papers: $total"
echo "  Location: $BASE_DIR"
echo "=========================================="
