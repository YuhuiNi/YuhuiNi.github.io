---
layout: post
title: Intro to Bone Age Prediction Algorithm(Intern at Infervision)
---

In this article, I'd like to give a brief introduction of my bone age prediction algorithm at Infervision. Specially, the algorithm mainly consists of four parts:

* Resnet. The goal of the algorithm is similar to other object detection algorithms, so it uses state-of-the-art backbone newwork, i.e. Resnet to extract the feature map of orginal image and takes that as an input to the following feature pyramid network(FPN) and bone age key points prediction.
* FPN(feature pyramid network). This part accepts different feature maps of bottleneck of Resnet and then forms a feature pyramid. Instead of predicting result at every level, we only predict at finest level and uses that as input of location predict and score prediction.
* Prediction of bone age key point location. After we have the convolution layer of FPN, we use focal loss function instead of traditional cross entropy loss function at last layer, which achieves a better result.
* Prediction of bone age score. The inputs of this part are final convolution layers in FPN and two Fully connected layers in last two bottleneck feature maps of Resnet and we also use focal loss as our loss function.

#### 1.Resnet

**Main idea**: Residual reflects that the neural network learns the difference rather than the absolute mapping. By learning the mapping relative to the original deviation, i.e. the difference from the identity part, it's easier for deep network to learn parameters. The existence of shortcut connection also effectively avoids gradient vanishing in backpropagation.

![resnet](https://github.com/YuhuiNi/YuhuiNi.github.io/raw/master/img/resnet1.png){:width="450"}

**Basic structure**

Resnet consits of two basic units:

Default backbone network of our bone age prediction algorithm is resnet50, which uses basic unit in right hand side. We can change the number of channels via 1x1 conv layer. The stride of first 3x3 conv layer is 2(so as first shortcut layer) while stride of remaining 3x3 conv layer is 1.

![fpn](https://github.com/YuhuiNi/YuhuiNi.github.io/raw/master/img/network_structure.png){:width="600"}


#### 2.FPN

Feature pyramind network's inputs are feature maps of bottleneck in Resnet. Every feature map is added to next level feature map after up sampling from top to bottom and thus each layer has different resolution and semantic features. Thus, different feature layer can be used to detect objects with different sizes. At the same time, FPN is just like putting additional connections on original Resnet, so it does not cost extra time and computation in practice.

![fpn](https://github.com/YuhuiNi/YuhuiNi.github.io/raw/master/img/Resnet%2Bfpn.png){:width="600"}

We notice that the size of ground truth in our task is fixed, so we only predict location and score on finest level feature map(i.e. the bottom layer), which is different from original FPN.

#### 3.Prediction of bone age key point location

**1.Focal loss**

I'd like to talk about focal loss first.

![focal loss](https://github.com/YuhuiNi/YuhuiNi.github.io/raw/master/img/focal_loss.png){:width="450"}

The author proposes a novel loss function,Focal Loss, which is achieved by adding a factor \\((1-p_t)^{\gamma}\\) to the standard cross entropy criterion. Setting \\(\gamma >0 \\)reduces the relative loss for well-classified examples (\\(p_t>.5\\)), putting more focus on hard, misclassified examples.

In our bone age predict algorithm, we use the default setting \\(\gamma=2\\). With this, FL can effectively discount the effect of easy negatives, focusing all attention on the hard negative examples.

![fc_plot](https://github.com/YuhuiNi/YuhuiNi.github.io/raw/master/img/fc_plot.png){:width="500"}

**2.key point location prediction**


![location](https://github.com/YuhuiNi/YuhuiNi.github.io/raw/master/img/boneage_predict.png){:width="660"}

In our original dataset, every hand has 13 key points. Then we construct a small region and a large region whose centroids are key point. We can obtain a point location map by setting all values in small regions to be 1 and other to be 0. At the same time, we can obtain a region location map by repeating same process for large regions. 

Why we want to construct point map and region map? There are two reasons:

* Identify the location of key point.
* Restrict our predictions within large regions and achieve a better result by setting small regions.

Our final predicton layer is **13\*512\*512**. Each layer is responsible for predict the location of one key point. Combined with previous point location map and region location map, we can already compute loss using focal loss function.

This algorithm has another part to ensure we predict 13 key points as a whole. Use max function along channel and takes that    as the probability of being key point. 

#### 4.Prediction of bone age score

![location](https://github.com/YuhuiNi/YuhuiNi.github.io/raw/master/img/score%20diagram.png){:width="660"}

Similar to **Part 3**, we first obtain a **9\*512\*512** layer. Takes the max value along channel and we get a predicted score degree and L1 loss(compared with true label). At the same time, we choose the predicted layer corresponding to true label(for example, we choose 5th layer if true label is 5) and combine it with point location map to compute focal loss.

In order to speed up the convergence rate, we also add two FC layers at last two bottleneck in Resnet. Those two **13\*9** layers are also combined with true label to compute focal loss.









