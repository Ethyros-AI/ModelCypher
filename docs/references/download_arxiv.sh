#!/bin/bash
# Script to download reference PDFs with clear naming (mostly arXiv)
# Usage: ./download_arxiv.sh

cd "$(dirname "$0")/arxiv"

# Array of papers: SOURCE|Filename
# SOURCE is either an arXiv ID (e.g. 2209.04836, gr-qc/9310026) or a direct PDF URL.
papers=(
    "1706.03741|Christiano_2017_Deep_RL_Human_Preferences"
    "1905.00414|Kornblith_2019_CKA_Neural_Similarity"
    "2004.06093|Naitzat_2020_Topology_Deep_Neural_Networks"
    "2106.09685|Hu_2022_LoRA_Low_Rank_Adaptation"
    "2107.07511|Angelopoulos_2021_Conformal_Prediction_Intro"
    "2203.02155|Ouyang_2022_InstructGPT_RLHF"
    "2209.04836|Ainsworth_2023_Git_ReBasin"
    "2209.11895|Olsson_2022_Induction_Heads_ICL"
    "2210.01892|Scherlis_2022_Polysemanticity_Capacity"
    "2211.00593|Wang_2022_IOI_Circuit_GPT2"
    "2212.04089|Ilharco_2023_Task_Arithmetic"
    "2212.08073|Bai_2022_Constitutional_AI"
    "2303.08112|Belrose_2023_Tuned_Lens"
    "2305.14314|Dettmers_2023_QLoRA"
    "2305.18290|Rafailov_2023_DPO"
    "2306.01708|Yadav_2023_TIES_Merging"
    "2306.03341|Li_2023_Inference_Time_Intervention"
    "2308.10248|Turner_2024_Activation_Addition"
    "2309.16042|Zhang_2024_Activation_Patching"
    "2310.01405|Zou_2023_Representation_Engineering"
    "2310.10631|Deletang_2024_Language_Compression"
    "2310.12036|Azar_2024_IPO"
    "2311.03348|Shah_2023_Persona_Modulation_Jailbreaks"
    "2312.06674|Inan_2023_Llama_Guard"
    "2402.04249|Mazeika_2024_HarmBench"
    "2404.02151|Andriushchenko_2024_Adaptive_Jailbreaks"
    "2404.02954|Robinson_2024_Token_Embeddings_Manifold"
    "2405.07987|Huh_2024_Platonic_Representation"
    "2405.14734|Meng_2024_SimPO"
    "2406.01171|Chen_2024_Two_Tales_Persona"
    "2406.04313|Zou_2024_Circuit_Breakers"
    "2406.11717|Arditi_2024_Refusal_Single_Direction"
    "2406.12411|DiSipio_2024_Information_Geometry_LLM"
    "2406.15927|Kossen_2024_Semantic_Entropy_Probes"
    "2406.16323|Liu_2024_MKA_Pruning_Merging"
    "2410.02106|Shape_Happens_2024"
    "2410.02355|Fang_2025_AlphaEdit"
    "2410.08993|TokenSpace_2024_Structure"
    "2412.00081|TSV_2025_Task_Singular_Vectors"
    "2502.16570|Ali_2025_Entropy_Lens"
    "2502.18821|CAMEx_2025_Fisher_Information"
    "2503.00555|Huang_2025_Safety_Tax"
    "2503.08099|WUDI_2025_Task_Vector_Subspaces"
    "2505.24445|Safety_Polytope_2025"
    "2506.03523|Li_2025_TokAlign"
    "2506.06609|Model_Stitching_2025"
    "2507.17075|Xue_2025_LoRA_Safety_Alignment"
    "2507.21509|Chen_2025_Persona_Vectors_Anthropic"
    "2509.21413|NUFILT_2025_Null_Space_Projection"
    "2510.11278|ENIGMA_2025_Geometry_Reasoning"
    "2512.11391|Niu_2025_NSPO_Null_Space"
    "gr-qc/9310026|tHooft_1993_Dimensional_Reduction"
    "https://raw.githubusercontent.com/mlresearch/v235/main/assets/bertolotti24a/bertolotti24a.pdf|Bertolotti_2024_Tying_Embeddings"
)

total=${#papers[@]}
count=0

echo "Downloading $total reference PDFs..."

for paper in "${papers[@]}"; do
    IFS='|' read -r source filename <<< "$paper"
    count=$((count + 1))
    
    if [ ! -f "${filename}.pdf" ]; then
        echo "[$count/$total] Downloading: $filename"
        if [[ "$source" == http* ]]; then
            url="$source"
        else
            url="https://arxiv.org/pdf/${source}.pdf"
        fi
        curl -sL -o "${filename}.pdf" "$url"
        sleep 0.5  # Be nice to arXiv servers
    else
        echo "[$count/$total] Skipping (exists): $filename"
    fi
done

echo ""
echo "Download complete!"
ls -lh *.pdf 2>/dev/null | wc -l
echo "PDFs downloaded"
