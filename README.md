Official PyTorch implementation of NeoNeXt models.

[NeoNeXt: Novel neural network operator and architecture based on the patch-wise matrix multiplications](https://arxiv.org/abs/2403.11251).
Vladimir Korviakov, Denis Koposov

## Environment

-- CUDA 12.1
-- PyTorch 2.3.1
-- torchvision 0.18.1

## Run

The example of training command:
```bash
pip install -r requirements.txt
bash scripts/run_neonext_imagenet_local.sh
```

## Implementation details

Models are defined in file ptvision/models/neonext/neonxet.py.
Currently there are several models considered as best (but not finally):
* NeoNeXt-T
* NeoNeXt-S
* NeoNeXt-B
* NeoNeXt-L

(Where the number of NeoNeXt-X means nothing but the version).

Each model has different number of blocks in each stage and different number of channels.

The block of NeoNeXt looks similar to ConvNeXt, but NeoCell is used instead of the depthwise convolution.

Also NeoCells are used for down-sampling in stem, between the stages and after the final feature map.

NeoCell implementation can be found in the file ptvision/models/neonext/neonxet_utils.py:
Otimized PyTorch implementation in C++ API of NeoCell functions can be found in file ptvision/models/neonext/csrc/neocell.cpp.

Given input of shape NxCxHxW NeoCell performs channel-wise matrix multiplications using two trainable matrices A and B (pair of matrices for each channel): Y=A\*X\*B.

All input channels are splitted to several groups of "channel" number (may be different for each group).

Each group is processed by matrices of the same size.

If "kernel" parameter is set, both A and B matrices are squared matrices of size kernel and spatial size of the data is not changed.

If "h_in", "h_out", "w_in", "w_out" parameters are set, A has size h_out\*h_in and B has size w_in\*w_out and spatial size of the data can be changed (both increased or decreased).

If the "shift" is set (non zero) then all channels for this kernal are splitted to "kernel" sub-groups. And blocks in block-diagonal matrix in each next sub-group will be shifted by 1 in horizontal and vertical directions. The blocks are cycled and parts of kernels can be used in the lower-right and upper left corners of the block-diagonal matrix. The "shift" is supported only for squared matrices.

## ImageNet-1k results

| Model     | res | #params | GFLOPs | acc@1 |
|-----------|-----|---------|--------|-------|
| NeoNeXt-T | 224 | 27.5M   |  4.4   |  TBD  |
| NeoNeXt-S | 224 | 49.7M   |  8.6   |  TBD  |
| NeoNeXt-B | 224 | 87.4    |  15.2  |  TBD  |
| NeoNeXt-T | 384 | 27.5M   |  13.3  |  TBD  |
| NeoNeXt-S | 384 | 49.7M   |  25.7  |  TBD  |
| NeoNeXt-B | 384 | 87.4    |  45.2  |  TBD  |
| NeoNeXt-L | 384 | 194.4   |  TBD   |  TBD  |

## Citations

```
@misc{korviakov2024neonext,
      title={NeoNeXt: Novel neural network operator and architecture based on the patch-wise matrix multiplications}, 
      author={Vladimir Korviakov and Denis Koposov},
      year={2024},
      eprint={2403.11251},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```