import glob
import os
import torch
import json
import numpy as np
from PIL import Image

from torchmetrics.image import PeakSignalNoiseRatio

# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

psnr_fn = PeakSignalNoiseRatio()

def evaluate_per_sample(pred, gt):
    psnr_value = psnr_fn(pred, gt)
    return psnr_value, -10, -10


def process_folder(pred_dir, gt_dir, factor):
    pred_paths = glob.glob(os.path.join(pred_dir, f'renders_{factor}/test/rgb', '*.jpg'))
    gt_paths = [os.path.join(gt_dir, f'images_{factor}', os.path.basename(x).replace('.jpg', '.JPG')) for x in pred_paths]

    gt_paths = [x.replace(f'images_1', 'images') for x in gt_paths]

    metric_values = []

    for pred_path, gt_path in zip(pred_paths, gt_paths):
        pred_image = Image.open(pred_path).convert('RGB')
        gt_image = Image.open(gt_path).convert('RGB')

        pred_image = torch.from_numpy(np.array(pred_image))
        gt_image = torch.from_numpy(np.array(gt_image))

        pred_image = pred_image.permute(2, 0, 1)[None, ...]
        gt_image = gt_image.permute(2, 0, 1)[None, ...]

        gt_image = torch.nn.functional.interpolate(gt_image, pred_image.shape[-2:])

        psnr_value, ssim_value, lpips_value = evaluate_per_sample(pred_image, gt_image)

        metric_values.append([psnr_value, ssim_value, lpips_value])
    
    metric_values = torch.tensor(metric_values) # (n, 3)

    return metric_values.mean(dim=0)


base_dir = '/home/space/exps/ns_dila_exps/dila_fixop'
dataset_dir = '/home/luzhan/Datasets/nerf_360_v2'
factors = [1, 2, 4, 8]

# /home/space/exps/ns_dila_exps/dila/dila_0.08/bicycle/splatfacto/0/renders_8/test/rgb

exp_dirs = glob.glob(os.path.join(base_dir, 'dila*'))
exp_dirs = sorted(exp_dirs, key=lambda x: float(x.split('_')[-1]))

# scenes = os.listdir(dataset_dir)
# scenes = sorted(scenes)

scenes = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]

fp = open('eval/collected_metrics.csv', 'w+')
fp.write('Method,' + ','.join([s for s in scenes for _ in range(4)]) + '\n')
fp.write('Res,' + ','.join([x for _ in scenes for x in ['1', '1/2', '1/4', '1/8']]) + '\n')

for exp_dir in exp_dirs:
    line = f'{os.path.basename(exp_dir)},'

    for scene in scenes:
        line_perscene = ''

        for factor in factors:
            exp_path = os.path.join(exp_dir, scene, f'splatfacto/0')
            gt_path = os.path.join(dataset_dir, scene)

            metric_files = glob.glob(os.path.join(exp_path, f'renders_{factor}', f'metric_*.json'))
            if len(metric_files) >= 1:
                result = os.path.basename(metric_files[0])[:-5].split('_')[1:]
                result = torch.tensor([float(x) for x in result])
            else:
                try:
                    result = process_folder(pred_dir=exp_path, gt_dir=gt_path, factor=factor)
                    save_path = os.path.join(exp_path, f'renders_{factor}', f'metric_{result[0]}_{result[1]}_{result[2]}.json')
                    with open(save_path, 'w+') as f:
                        json.dump({}, f)
                except:
                    result = torch.tensor([-10, -10, -10])
            
            print(os.path.basename(exp_dir), scene, factor, result)

            line_perscene += f'{result[0]},'
        
        line += line_perscene

    line = line[:-1] + '\n'
    fp.write(line)
    fp.flush()

fp.close()
