export CUDA_VISIBLE_DEVICES=0

data_dir=/home/luzhan/Datasets/nerf_360_v2
old_exp_dir=/home/space/exps/ns_dila_exps/base_15k
exp_dir=/home/space/exps/ns_dila_exps/dila_fixop

# scenes=("bicycle" "bonsai" "counter" "flowers" "garden" "stump" "treehill" "kitchen" "room")
scenes=("bicycle" "bonsai" "counter" "flowers" "garden")
dilation_factors=(0.02 0.03 0.05)
downscale_factors=(1 2 4 8)

for dilation_factor in ${dilation_factors[@]}; do
    for scene in ${scenes[@]}; do
        ns-train splatfacto \
            --pipeline.model.scaleup-point-post-densification True \
            --pipeline.model.scaleup-factor ${dilation_factor} \
            --pipeline.model.scaleup-by-scale True \
            --pipeline.model.scaleup-bias 0.2 \
            --pipeline.model.scaleup-interval 1000 \
            --pipeline.model.scaleup_th_mode top97 \
            --pipeline.model.scaleup_annealing False \
            --pipeline.model.continue_cull_post_densification False \
            --pipeline.model.rasterize-mode antialiased \
            --pipeline.model.cull_alpha_thresh 0.005 \
            --pipeline.model.num-downscales 0 \
            --vis wandb \
            --pipeline.model.background-color black \
            --steps-per-save 2000 \
            --save-only-latest-checkpoint False \
            --project-name mip_recon \
            --output-dir ${exp_dir}/dila_${dilation_factor} \
            --experiment-name ${scene} \
            --timestamp 0 \
            --max-num-iterations 15000 \
            --load-checkpoint ${old_exp_dir}/base_15k/${scene}/splatfacto/0/nerfstudio_models/step-000014999.ckpt \
            colmap \
            --data ${data_dir}/${scene} \
            --downscale-factor 8 \
            --colmap-path sparse/0 \
            --downscale-rounding-mode round

        for downscale_factor in ${downscale_factors[@]}; do
            ns-render dataset \
                --load-config ${exp_dir}/dila_${dilation_factor}/${scene}/splatfacto/0/config.yml \
                --downscale-factor ${downscale_factor} \
                --rendered-output-names rgb \
                --output-path ${exp_dir}/dila_${dilation_factor}/${scene}/splatfacto/0/renders_${downscale_factor}
        done
        
    done
done


exp_dir=/home/space/exps/ns_dila_exps/dila_fixop_annealing
dilation_factors=(0.01 0.02 0.05 0.1 0.2)
for dilation_factor in ${dilation_factors[@]}; do
    for scene in ${scenes[@]}; do
        ns-train splatfacto \
            --pipeline.model.scaleup-point-post-densification True \
            --pipeline.model.scaleup-factor ${dilation_factor} \
            --pipeline.model.scaleup-by-scale True \
            --pipeline.model.scaleup-bias 0.2 \
            --pipeline.model.scaleup-interval 1000 \
            --pipeline.model.scaleup_th_mode top97 \
            --pipeline.model.scaleup_annealing True \
            --pipeline.model.continue_cull_post_densification False \
            --pipeline.model.rasterize-mode antialiased \
            --pipeline.model.cull_alpha_thresh 0.005 \
            --pipeline.model.num-downscales 0 \
            --vis wandb \
            --pipeline.model.background-color black \
            --steps-per-save 2000 \
            --save-only-latest-checkpoint False \
            --project-name mip_recon \
            --output-dir ${exp_dir}/dila_${dilation_factor} \
            --experiment-name ${scene} \
            --timestamp 0 \
            --max-num-iterations 15000 \
            --load-checkpoint ${old_exp_dir}/base_15k/${scene}/splatfacto/0/nerfstudio_models/step-000014999.ckpt \
            colmap \
            --data ${data_dir}/${scene} \
            --downscale-factor 8 \
            --colmap-path sparse/0 \
            --downscale-rounding-mode round

        for downscale_factor in ${downscale_factors[@]}; do
            ns-render dataset \
                --load-config ${exp_dir}/dila_${dilation_factor}/${scene}/splatfacto/0/config.yml \
                --downscale-factor ${downscale_factor} \
                --rendered-output-names rgb \
                --output-path ${exp_dir}/dila_${dilation_factor}/${scene}/splatfacto/0/renders_${downscale_factor}
        done
        
    done
done



exp_dir=/home/space/exps/ns_dila_exps/dila_fixop_annealing_thmean
dilation_factors=(0.01 0.02 0.05 0.1 0.2)
for dilation_factor in ${dilation_factors[@]}; do
    for scene in ${scenes[@]}; do
        ns-train splatfacto \
            --pipeline.model.scaleup-point-post-densification True \
            --pipeline.model.scaleup-factor ${dilation_factor} \
            --pipeline.model.scaleup-by-scale True \
            --pipeline.model.scaleup-bias 0.2 \
            --pipeline.model.scaleup-interval 1000 \
            --pipeline.model.scaleup_th_mode mean \
            --pipeline.model.scaleup_annealing True \
            --pipeline.model.continue_cull_post_densification False \
            --pipeline.model.rasterize-mode antialiased \
            --pipeline.model.cull_alpha_thresh 0.005 \
            --pipeline.model.num-downscales 0 \
            --vis wandb \
            --pipeline.model.background-color black \
            --steps-per-save 2000 \
            --save-only-latest-checkpoint False \
            --project-name mip_recon \
            --output-dir ${exp_dir}/dila_${dilation_factor} \
            --experiment-name ${scene} \
            --timestamp 0 \
            --max-num-iterations 15000 \
            --load-checkpoint ${old_exp_dir}/base_15k/${scene}/splatfacto/0/nerfstudio_models/step-000014999.ckpt \
            colmap \
            --data ${data_dir}/${scene} \
            --downscale-factor 8 \
            --colmap-path sparse/0 \
            --downscale-rounding-mode round

        for downscale_factor in ${downscale_factors[@]}; do
            ns-render dataset \
                --load-config ${exp_dir}/dila_${dilation_factor}/${scene}/splatfacto/0/config.yml \
                --downscale-factor ${downscale_factor} \
                --rendered-output-names rgb \
                --output-path ${exp_dir}/dila_${dilation_factor}/${scene}/splatfacto/0/renders_${downscale_factor}
        done
        
    done
done