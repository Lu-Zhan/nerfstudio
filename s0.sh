
data_dir=/home/luzhan/Datasets/nerf_360_v2
exp_dir=outputs/mip_dilation_extra

# scenes=("bicycle" "bonsai" "counter" "flowers" "garden" "stump" "treehill" "kitchen" "room")
# dilation_factors=(1 0.9 0.8 0.5 0.2)
scenes=("bicycle")
dilation_factors=(0.9)

for dilation_factor in ${dilation_factors[@]}; do
    for scene in ${scenes[@]}; do
        ns-train splatfacto \
            --pipeline.model.scaleup-point-post-densification True \
            --pipeline.model.scaleup-factor ${dilation_factor} \
            --pipeline.model.scaleup-by-scale True \
            --pipeline.model.scaleup-bias 0.2 \
            --pipeline.model.scaleup-interval 1000 \
            --vis wandb \
            --project-name mip_recon \
            --output-dir ${exp_dir}/dila_${dilation_factor} \
            --experiment-name ${scene} \
            --timestamp 0 \
            colmap \
            --data ${data_dir}/${scene} \
            --downscale-factor 8 \
            --colmap-path sparse/0 \
            --downscale-rounding-mode round
    done
done

ns-render dataset --load-config outputs/mip_dilation_extra/dila_0.9/bicycle/splatfacto/0/config.yml --downscale-factor 4 --rendered-output-names gt-rgb rgb  --output-path renders_4