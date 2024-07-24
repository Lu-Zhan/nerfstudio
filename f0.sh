export CUDA_VISIBLE_DEVICES=1

data_dir=/home/luzhan/Datasets/nerf_360_v2
scenes=("bicycle")
exp_dir=/home/space/exps/ns_dila_exps/freq_viewer_progress
dilation_factors=(0)

for dilation_factor in ${dilation_factors[@]}; do
    for scene in ${scenes[@]}; do
        ns-train splatfacto \
            --pipeline.model.scaleup-point-post-densification True \
            --pipeline.model.scaleup-factor ${dilation_factor} \
            --pipeline.model.scaleup-by-scale True \
            --pipeline.model.scaleup_by_grad True \
            --pipeline.model.scaleup-bias 0.2 \
            --pipeline.model.scaleup-interval 1000 \
            --pipeline.model.scaleup_th_mode top97 \
            --pipeline.model.scaleup_annealing False \
            --pipeline.model.continue_cull_post_densification False \
            --pipeline.model.rasterize-mode antialiased \
            --pipeline.model.cull_alpha_thresh 0.005 \
            --pipeline.model.num-downscales 0 \
            --vis viewer \
            --viewer.quit-on-train-completion True \
            --pipeline.model.background-color black \
            --save-only-latest-checkpoint True \
            --project-name mip_recon \
            --output-dir ${exp_dir}/dila_${dilation_factor} \
            --experiment-name ${scene} \
            --timestamp 0 \
            --max-num-iterations 30000 \
            colmap \
            --data ${data_dir}/${scene} \
            --downscale-factor 8 \
            --colmap-path sparse/0 \
            --downscale-rounding-mode round
    done
done