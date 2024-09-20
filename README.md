# Taming 3DGS
## High-Quality Radiance Fields with Limited Resources
Saswat Subhajyoti Mallick*, Rahul Goel*, Bernhard Kerbl, Francisco Vicente Carrasco, Markus Steinberger, Fernando De La Torre (* indicates equal contribution)

[Webpage](https://humansensinglab.github.io/taming-3dgs/) | [Arxiv](https://arxiv.org/abs/2406.15643)


**Abstract:** *3D Gaussian Splatting (3DGS) has transformed novel-view synthesis with its fast, interpretable, and high-fidelity rendering. However, its resource requirements limit its usability. Especially on constrained devices, training performance degrades quickly and often cannot complete due to excessive memory consumption of the model. The method converges with an indefinite number of Gaussians—many of them redundant—making rendering unnecessarily slow and preventing its usage in downstream tasks that expect fixed-size inputs. To address these issues, we tackle the challenges of training and rendering 3DGS models on a budget. We use a guided, purely constructive densification process that steers densification toward Gaussians that raise the reconstruction quality. Model size continuously increases in a controlled manner towards an exact budget, using score-based densification of Gaussians with training-time priors that measure their contribution. We further address training speed obstacles: following a careful analysis of 3DGS’ original pipeline, we derive faster, numerically equivalent solutions for gradient computation and attribute updates, including an alternative parallelization for efficient backpropagation. We also propose quality-preserving approximations where suitable to reduce training time even further. Taken together, these enhancements yield a robust, scalable solution with reduced training times, lower compute and memory requirements, and high quality. Our evaluation shows that in a budgeted setting, we obtain competitive quality metrics with 3DGS while achieving a 4–5× reduction in both model size and training time. With more generous budgets, our measured quality surpasses theirs. These advances open the door for novelview synthesis in constrained environments, e.g., mobile devices.*

---

Currently, this repository only contains the performance optimizations as drop-in replacements for the original implementation. It **does not** provide the changes to the densification scheme. The completely code will be released in late November 2024 prior to SIGGRAPH Asia 2024 conference dates.

**All the performance optimizations are released under the MIT License.**

Please refer to the [Inria repository](https://github.com/graphdeco-inria/gaussian-splatting) for complete instructions.

## Training time comparison
<a href="https://www.inria.fr/"><img height="400" src="assets/times_compare.png"> </a>

## Sparse Adam Optimizer
Sparse adam is an alternate implementation for the Adam optimizer that applies gradients only to the gaussians that are visible. Note that this can change the training behaviour, whereas all other changes don't affect the numerical results.

## Using the sparse adam optimizer
To use the sparse adam optimizer, the following argument can be appended to the training script
```bash
--optimizer_type sparse_adam
```

## Quality comparison
Following is the difference in quality between the original 3DGS algorithm and our sparse adam implementation.

<table><thead>
  <tr>
    <th rowspan="2"></th>
    <th colspan="2">SSIM</th>
    <th colspan="2">PSNR</th>
    <th colspan="2">LPIPS</th>
  </tr>
  <tr>
    <th>3DGS</th>
    <th style="text-align: center"> Ours <br> (sparse adam)</th>
    <th >3DGS</th>
    <th style="text-align: center"> Ours <br> (sparse adam)</th>
    <th>3DGS</th>
    <th style="text-align: center"> Ours <br> (sparse adam)</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Mip-NeRF360</td>
    <td align="center">0.815</td>
    <td align="center">0.807</td>
    <td align="center">27.46</td>
    <td align="center">27.33</td>
    <td align="center">0.215</td>
    <td align="center">0.229</td>
  </tr>
  <tr>
    <td>Tanks&amp;Temples</td>
    <td align="center">0.847</td>
    <td align="center">0.846</td>
    <td align="center">23.64</td>
    <td align="center">23.56</td>
    <td align="center">0.176</td>
    <td align="center">0.180</td>
  </tr>
  <tr>
    <td>Deep Blending</td>
    <td align="center">0.904</td>
    <td align="center">0.907</td>
    <td align="center">29.64</td>
    <td align="center">29.69</td>
    <td align="center">0.243</td>
    <td align="center">0.247</td>
  </tr>
</tbody>
</table>
