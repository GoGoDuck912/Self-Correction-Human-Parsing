# Self Correction for Human Parsing

An out-of-box human parsing representation extracter. Also the 3rd LIP challenge winner solution!

![lip-visualization](./img/lip-visualization.jpg)

At this time, we provide the trained models on three popular human parsing datasets that achieve the state-of-the-art performance. We hope our work could serve as a basic human parsing representation extracter and faciliate your own tasks.

## TODO List

- [x] Inference code on three popular single person human parsing datasets.
- [ ] Training code
- [ ] Inference code on multi-person and video human parsing datasets.

Coming Soon! Stay tuned!

## Requirements

```
Python >= 3.5, PyTorch >= 0.4
```

## Pretrained models

The easist way to get started is to use our trained SCHP models on your own images to extract human parsing representations. Here we provided trained models on three popular datasets. Theses three datasets have different label system, you can choose the best one to fit on your own task.

**LIP** ([exp-schp-201908261155-lip.pth](https://drive.google.com/file/d/1ZrTiadzAOM332d896fw7JZQ2lWALedDB/view?usp=sharing))

* mIoU on LIP validation: **59.36 %**.

* LIP is the largest single person human parsing dataset with 50000+ images. This dataset focus more on the complicated real scenarios. LIP has 20 labels, including 'Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe'.

**ATR** ([exp-schp-201908301523-atr.pth](https://drive.google.com/file/d/1klCtqx51orBkFKdkvYwM4qao_vEFbJ_z/view?usp=sharing))

* mIoU on ATR test: **82.29%**.

* ATR is a large single person human parsing dataset with 17000+ images. This dataset focus more on fashion AI. ATR has 18 labels, including 'Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt', 'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'.

**Pascal-Person-Part** ([exp-schp-201908270938-pascal-person-part.pth](https://drive.google.com/file/d/13ph1AloYNiC4DIGOyCLZdmA08tP9OeGu/view?usp=sharing))

* mIoU on Pascal-Person-Part validation: **71.46** %.

* Pascal Person Part is a tiny single person human parsing dataset with 3000+ images. This dataset focus more on body parts segmentation. Pascal Person Part has 7 labels, including 'Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'.

Choose one and have fun on your own task!

## Inference

To extract the human parsing representation, simply put your own image in the `Input_Directory`, download a pretrained model and run the following command. The output images with the same file name will be saved in `Output_Directory`

```
python evaluate.py --dataset Dataset --restore-weight Checkpoint_Path --input Input_Directory --output Output_Directory
```

The `Dataset` command has three options, including 'lip', 'atr' and 'pascal'. Note each pixel in the output images denotes the predicted label number. The output images have the same size as the input ones. To better visualization, we put a palette with the output images. We suggest you to read the image with `PIL`.

If you need not only the final parsing image, but also a feature map representation. Add `--logits` command to save the output feature map. This feature map is the logits before softmax layer with the dimension of HxWxC.


## Visualization

* Source Image.
![demo](./input/demo.jpg)

* LIP Parsing Result.
![demo-lip](./output/demo_lip.png)

* ATR Parsing Result.
![demo-atr](./output/demo_atr.png)

* Pascal-Person-Part Parsing Result.
![demo-pascal](./output/demo_pascal.png)


## Related

There is also a [PaddlePaddle](https://github.com/PaddlePaddle/PaddleSeg/tree/master/contrib/ACE2P) Implementation.
This implementation is the version that we submitted to the 3rd LIP Challenge.