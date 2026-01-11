#!/bin/bash
# Script to download reference PDFs with clear naming (mostly arXiv)
# Usage: ./download_arxiv.sh

cd "$(dirname "$0")/arxiv"

# Array of papers: SOURCE|Filename
# SOURCE is either an arXiv ID (e.g. 2209.04836, gr-qc/9310026) or a direct PDF URL.
papers=(
    "0711.0189|Luxburg_2007_Tutorial_Spectral_Clustering"
    "1310.0425|Fefferman_2013_Testing_Manifold_Hypothesis"
    "1503.02406|Tishby_2015_Deep_Learning_Information_Bottleneck_Principle"
    "1503.05671|Martens_2015_Optimizing_Neural_Networks_Kroneckerfactored_Approximate_Curvature"
    "1602.07868|Salimans_2016_Weight_Normalization_Simple_Reparameterization_Accelerate_Training"
    "1605.09096|Hamilton_2016_Diachronic_Word_Embeddings_Reveal_Statistical_Laws"
    "1706.03741|Christiano_2017_Deep_RL_Human_Preferences"
    "1706.04599|Guo_2017_Calibration_Modern_Neural_Networks"
    "1803.00567|Peyre_2018_Computational_Optimal_Transport"
    "1809.00013|AlvarezMelis_2018_GromovWasserstein_Alignment_Word_Embedding_Spaces"
    "1811.02834|Vayer_2018_Fused_GromovWasserstein_distance_structured_objects_theoretical"
    "1905.00414|Kornblith_2019_CKA_Neural_Similarity"
    "1905.09418|Voita_2019_Analyzing_MultiHead_SelfAttention_Specialized_Heads_Do"
    "1905.12784|Ansuini_2019_Intrinsic_dimension_data_representations_deep_neural"
    "1910.05653|Singh_2019_Model_Fusion_Optimal_Transport"
    "2003.00335|Lou_2020_Differentiating_through_Frechet_Mean"
    "2004.06093|Naitzat_2020_Topology_Deep_Neural_Networks"
    "2012.13255|Aghajanyan_2021_Intrinsic_Dimensionality_Fine_Tuning"
    "2104.08894|Pope_2021_Intrinsic_Dimension_Images_Impact_Learning"
    "2106.09685|Hu_2022_LoRA_Low_Rank_Adaptation"
    "2107.07511|Angelopoulos_2021_Conformal_Prediction_Intro"
    "2108.01661|Ding_2021_Grounding_Representation_Similarity_Statistical_Testing"
    "2111.09832|Matena_2021_Merging_Models_FisherWeighted_Averaging"
    "2203.02155|Ouyang_2022_InstructGPT_RLHF"
    "2209.04836|Ainsworth_2023_Git_ReBasin"
    "2209.11895|Olsson_2022_Induction_Heads_ICL"
    "2209.15430|Moschella_2022_Relative_representations_enable_zeroshot_latent_space"
    "https://www.nature.com/articles/s41598-022-20991-1.pdf|Denti_2022_GRIDE_Generalized_Ratios_Intrinsic_Dimension"
    "2210.01892|Scherlis_2022_Polysemanticity_Capacity"
    "2211.00593|Wang_2022_IOI_Circuit_GPT2"
    "2212.04089|Ilharco_2023_Task_Arithmetic"
    "2212.08073|Bai_2022_Constitutional_AI"
    "2303.08112|Belrose_2023_Tuned_Lens"
    "2305.06329|Klabunde_2023_Similarity_Neural_Network_Models_Survey_Functional"
    "2305.14314|Dettmers_2023_QLoRA"
    "2305.18290|Rafailov_2023_DPO"
    "2306.01708|Yadav_2023_TIES_Merging"
    "2306.03341|Li_2023_Inference_Time_Intervention"
    "2308.10248|Turner_2024_Activation_Addition"
    "2309.16042|Zhang_2024_Activation_Patching"
    "2310.01405|Zou_2023_Representation_Engineering"
    "2310.10631|Deletang_2024_Language_Compression"
    "2310.12036|Azar_2024_IPO"
    "2311.03099|Yu_2023_Language_Models_are_Super_Mario_Absorbing"
    "2311.03348|Shah_2023_Persona_Modulation_Jailbreaks"
    "2312.06674|Inan_2023_Llama_Guard"
    "2402.04249|Mazeika_2024_HarmBench"
    "2402.09353|Liu_2024_DoRA_WeightDecomposed_LowRank_Adaptation"
    "2403.13257|Goddard_2024_Arcees_MergeKit_Toolkit_Merging_Large_Language"
    "2404.02151|Andriushchenko_2024_Adaptive_Jailbreaks"
    "2404.02954|Robinson_2024_Token_Embeddings_Manifold"
    "2404.12917|Ricciardi_2024_R3L_Relative_Representations_Reinforcement_Learning"
    "2405.00492|Peeperkorn_2024_Is_Temperature_Creativity_Parameter_Large_Language"
    "2405.01012|Murphy_2024_Correcting_Biased_Centered_Kernel_Alignment_Measures"
    "2405.07987|Huh_2024_Platonic_Representation"
    "2405.14734|Meng_2024_SimPO"
    "2406.01171|Chen_2024_Two_Tales_Persona"
    "2406.04313|Zou_2024_Circuit_Breakers"
    "2406.11717|Arditi_2024_Refusal_Single_Direction"
    "2406.12411|DiSipio_2024_Information_Geometry_LLM"
    "2406.15812|Basile_2024_Intrinsic_Dimension_Correlation_uncovering_nonlinear_connections"
    "2406.15927|Kossen_2024_Semantic_Entropy_Probes"
    "2406.16323|Liu_2024_MKA_Pruning_Merging"
    "2407.21092|Yang_2024_Entropy_Thermodynamics_Geometrization_Language_Model"
    "2410.02106|Shape_Happens_2024"
    "2410.02355|Fang_2025_AlphaEdit"
    "2410.08993|TokenSpace_2024_Structure"
    "2412.00081|TSV_2025_Task_Singular_Vectors"
    "2412.10416|SuperMerge_2024_Gradient_Based_Model_Merging"
    "2501.08145|Hildebrandt_2025_Refusal_Behavior_Large_Language_Models_Nonlinear"
    "2502.02421|AIM_2025_Activation_Informed_Merging"
    "2502.15104|Chun_2025_Estimating_Neural_Representation_Alignment_Sparsely_Sampled"
    "2502.16570|Ali_2025_Entropy_Lens"
    "2502.18821|CAMEx_2025_Fisher_Information"
    "https://openreview.net/pdf?id=0fD3iIBhlV|Cheng_2025_HighDimensional_Abstraction_Phase_LMs"
    "2503.00555|Huang_2025_Safety_Tax"
    "2503.08099|WUDI_2025_Task_Vector_Subspaces"
    "2503.09774|GW_Feature_Alignment_2025_Model_Merging"
    "2505.24445|Safety_Polytope_2025"
    "2506.01034|Ruppik_2025_Local_Intrinsic_Dimensions_Contextual_LMs"
    "2506.01599|Yu_2025_Connecting_Neural_Models_Latent_Geometries_Relative"
    "2506.03523|Li_2025_TokAlign"
    "2506.06609|Model_Stitching_2025"
    "2507.01966|Shen_2025_Alignment_Brains_AI_Evidence_Convergent_Evolution"
    "2507.12380|Krahn_2025_Heat_Kernel_Goes_Topological"
    "2507.17075|Xue_2025_LoRA_Safety_Alignment"
    "2507.21509|Chen_2025_Persona_Vectors_Anthropic"
    "2508.21815|Hyrup_2025_Achieving_HilbertSchmidt_Independence_Renyi_Differential_Privacy"
    "2509.21413|NUFILT_2025_Null_Space_Projection"
    "2510.11278|ENIGMA_2025_Geometry_Reasoning"
    "2510.13406|Maystre_2025_When_Embedding_Models_Meet_Procrustes_Bounds"
    "2510.17072|Kim_2025_DFNN_Deep_Frechet_Neural_Network_Framework"
    "2510.24342|Chen_2025_Unified_Geometric_Space_Bridging_AI_Models"
    "2512.11391|Niu_2025_NSPO_Null_Space"
    "2512.16245|AlignMerge_2025_Alignment_Preserving_Merging"
    "2512.24880|Xie_2025_mHC_ManifoldConstrained_HyperConnections"
    "gr-qc/9310026|tHooft_1993_Dimensional_Reduction"
    "https://aclanthology.org/P19-1356.pdf|Jawahar_2019_BERT_Structure"
    "https://openreview.net/pdf?id=w9P4xFBQK7|Lobashev_2025_PRH_Information_Geometry"
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
