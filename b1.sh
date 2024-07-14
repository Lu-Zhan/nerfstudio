export CUDA_VISIBLE_DEVICES=1
data_dir=/home/luzhan/Datasets/nerf_360_v2
exp_dir=/home/space/exps/ns_dila_exps/base_15k

# scenes=("bicycle" "bonsai" "counter" "flowers" "garden" "stump" "treehill" "kitchen" "room")
scenes=("kitchen" "room")

for scene in ${scenes[@]}; do
    ns-train splatfacto \
        --pipeline.model.scaleup-point-post-densification False \
        --pipeline.model.scaleup-factor ${dilation_factor} \
        --pipeline.model.scaleup-by-scale True \
        --pipeline.model.scaleup-bias 0.2 \
        --pipeline.model.scaleup-interval 1000 \
        --pipeline.model.continue_cull_post_densification False \
        --pipeline.model.rasterize-mode antialiased \
        --pipeline.model.cull_alpha_thresh 0.005 \
        --pipeline.model.num-downscales 0 \
        --vis wandb \
        --pipeline.model.background-color black \
        --steps-per-save 4000 \
        --save-only-latest-checkpoint False \
        --project-name mip_recon \
        --output-dir ${exp_dir}/base_15k \
        --experiment-name ${scene} \
        --timestamp 0 \
        --max-num-iterations 15000 \
        colmap \
        --data ${data_dir}/${scene} \
        --downscale-factor 8 \
        --colmap-path sparse/0 \
        --downscale-rounding-mode round
done