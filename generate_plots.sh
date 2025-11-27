#!/bin/bash
# Example script for generating performance plots using performance.py


# Configuration
PROJECT="GL-MPC-RL"
MODEL_NAME="graceful-planet-22"
OUTPUT_DIR="figures/${PROJECT}/offline-rollout-tight-regions"
DT=300

# Example 1: Generate all plots and save them
python visualizations/performance.py \
    --project "${PROJECT}" \
    --model_name "${MODEL_NAME}" \
    --rlmpc_folders offline-rl-trajectory offline-rl-trajectory-tight-region  \
    --plot_all \
    --save \
    --output_dir "${OUTPUT_DIR}" \
    --dt ${DT} \
    --labels "RL" "MPC" "RL-MPC offline rollout" "RL-MPC offline rollout (tight regions)"

# # Example 3: Generate plots with custom RL-MPC folders
# echo "Example 3: Generating plots with custom RL-MPC folders..."
# python visualizations/performance.py \
#     --project "${PROJECT}" \
#     --model_name "${MODEL_NAME}" \
#     --rlmpc_folders terminal_constr terminal_constr_all \
#     --plot_all \
#     --save \
#     --output_dir "${OUTPUT_DIR}/custom"

# # Example 4: Generate plots with custom labels
# echo "Example 4: Generating plots with custom labels..."
# python visualizations/performance.py \
#     --project "${PROJECT}" \
#     --model_name "${MODEL_NAME}" \
#     --plot_controls \
#     --plot_states \
#     --save \
#     --output_dir "${OUTPUT_DIR}/custom_labels" \
#     --labels "Reinforcement Learning" "Model Predictive Control" "Hybrid Method 1" "Hybrid Method 2" "Hybrid Method 3"

# # Example 5: Show plots interactively (without saving)
# echo "Example 5: Showing plots interactively (comment out for non-interactive environments)..."
# # python visualizations/performance.py \
# #     --project "${PROJECT}" \
# #     --model_name "${MODEL_NAME}" \
# #     --plot_all \
# #     --show

# # Example 6: Generate and both save and show plots
# echo "Example 6: Generate, save, and show plots..."
# # python visualizations/performance.py \
# #     --project "${PROJECT}" \
# #     --model_name "${MODEL_NAME}" \
# #     --plot_all \
# #     --save \
# #     --show \
# #     --output_dir "${OUTPUT_DIR}/interactive"

echo "All examples completed!"
echo "Generated plots are in: ${OUTPUT_DIR}"

