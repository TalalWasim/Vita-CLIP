# Official repository for Vita-CLIP: Video and text adaptive CLIP via Multimodal Prompting [CVPR 2023]

[Syed Talal Wasim](https://talalwasim.github.io),
[Muzammal Naseer](https://muzammal-naseer.netlify.app/),
[Salman Khan](https://salman-h-khan.github.io),
[Fahad Shahbaz Khan](https://sites.google.com/view/fahadkhans/home),
[Mubarak Shah](https://www.crcv.ucf.edu/person/mubarak-shah/)

**[Paper Link](https://arxiv.org/abs/2304.03307)** 


> **Abstract:**
>*Adopting contrastive image-text pretrained models like CLIP towards video classification has gained attention due to its cost-effectiveness and competitive performance. However, recent works in this area face a trade-off. Finetuning the pretrained model to achieve strong supervised performance results in low zero-shot generalization. Similarly, freezing the backbone to retain zero-shot capability causes significant drop in supervised accuracy. Because of this, recent works in literature typically train separate models for supervised and zero-shot action recognition.
In this work, we propose a multimodal prompt learning scheme that works to balance the supervised and zero-shot performance under a single **unified** training. Our prompting approach on the vision side caters for three aspects: 1) Global **video-level** prompts to model the data distribution; 2) Local **frame-level** prompts to provide per-frame discriminative conditioning; and 3) a **summary prompt** to extract a condensed video representation. Additionally, we define a prompting scheme on the text side to augment the textual context.
Through this prompting scheme, we can achieve state-of-the-art zero-shot performance on Kinetics-600, HMDB51 and UCF101 while remaining competitive in the supervised setting. By keeping the pretrained backbone frozen, we optimize a much lower number of parameters and retain the existing general representation which helps achieve the strong zero-shot performance.*


<p align="center">
  <img alt="intro_image" src="figs/intro.png" width="300"/>
</p>


## Updates:

April 24 2023: Released supervised training code for Vita-CLIP (stay tuned for zeroshot evaluation scripts and pretrained models)


## Environment Setup
Refer to `requirements.txt` for installing all python dependencies. We use python 3.8.13 with pytorch 1.14.0. 


## Supervised Training

### Dataset Preparation

We download the official version of Kinetics-400 from [here](https://github.com/cvdfoundation/kinetics-dataset) and videos are resized using code [here](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics).

We expect that `--train_list_path` and `--val_list_path` command line arguments to be a data list file of the following format
```
<path_1>,<label_1>
<path_2>,<label_2>
...
<path_n>,<label_n>
```
where `<path_i>` points to a video file, and `<label_i>` is an integer between `0` and `num_classes - 1`.
`--num_classes` should also be specified in the command line argument.

Additionally, `<path_i>` might be a relative path when `--data_root` is specified, and the actual path will be
relative to the path passed as `--data_root`.

The class mappings in the open-source weights are provided at [Kinetics-400 class mappings](data/k400_class_mappings.json)

### Download Pretrained CLIP Checkpoint

Download the **[pretrained CLIP checkpoint](https://drive.google.com/file/d/17xSat9ZqL8p3RjpfTdqjxrBfcwZgZ2OE/view?usp=sharing)**  and place under the `pretrained directory`.

### Training Instruction

For supervised training on the Kinetics-400 dataset, use the train script in the `train_scripts` directory. Modify the `--train_list_path`, `--train_list_val` according to the data location and modify the `--backbone_path` according to location where the pretrained checkpoint was downloaded and stored.

## Zeroshot Evaluation

*Scripts for zeroshot evaluation will be released soon*

## Pretrained Models

Vita-CLIP-B checkpoint can be downloaded **[here](https://drive.google.com/file/d/1ArrbJnydQwY_mGa1DKyhs5s0SYdBhF2r/view?usp=sharing)**.


## Citation
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@inproceedings{wasim2023vitaclip,
    title={Vita-CLIP: Video and text adaptive CLIP via Multimodal Prompting}, 
    author={Syed Talal Wasim and Muzammal Naseer and Salman Khan and Fahad Shahbaz Khan and Mubarak Shah},
    booktitle={CVPR},
    year={2023}
  }
```


## Acknowledgements
Our code is based on [EVL](https://github.com/OpenGVLab/efficient-video-recognition) and [XCLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP) repositories. We thank the authors for releasing their code. If you use our model, please consider citing these works as well.