# GS-Medical

## Training

To train ... with customized hyper-parameters, please make changes in `arguments/endonerf/default.py`.

To train ..., run the following example command:

```
python train.py -s data/EndoNeRF/pulling_soft_tissues/ --expname endonerf/cutting
```

## Testing

For testing, we perform rendering and evaluation separately.

### Rendering

To run the following example command to render the images:

```
python render.py --model_path output/endonerf/cutting  --skip_train --skip_test --reconstruct_video
```

Please follow [EndoGaussian](https://github.com/yifliu3/EndoGaussian/tree/master) to skip rendering. Of note, you can also set `--reconstruct_train`, `--reconstruct_test`, and `--reconstruct_video` to reconstruct and save the `.ply` 3D point cloud of the rendered outputs for  `train`, `test` and`video` sets, respectively.

### Evaluation

To evaluate the reconstruction quality, run following command:

```
python metrics.py --model_path output/endonerf/cutting -p test
```

Note that you can set `-p video`, `-p test`, `-p train` to select the set for evaluation.