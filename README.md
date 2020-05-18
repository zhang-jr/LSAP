<!-- # Learning Sampling-Agnostic Perturbation for Video Action Classification -->

## Introduction

This is the source code and additional visualization examples of our LSAP, *learning Sampling-Agnostic Perturbations for Video Action Classification*.

**Motivation of Our Work**
<!-- Adversarial attack mainly focus on static image, but dose not deal with temporal varying inputs. An explict analysis of adversarial attack of temporal varying input on video undersatnding is still missing. Here, we attempt to investigate what properties are significant to generating adversarial examples for temporal varying inputs like video. The reason why we needed to generated unversarial perturbatio for a given video is that traditional adversarial attack methods only generate perturbation for a specific input, while a video recognition model can classify a video with any fixed input clips because of the inherent pattern distributed in the whole video. Therefore, the perturbations of videos should be valiable for any clips with different sampling strategy. -->
Intuitively, generating an adversarial example for a video is more difficult than for an image, since a video contains a sequence of images (frames) with strong temporal correlation. Directly applying the existing image-based adversarial attack methods to generate image-level adversarial examples for each frame in a video will inevitably neglect the  temporal correlation among frames and result in less effective video-level adversarial example.


**Insight of Our Work**

1. We investigate the problem of adversarial attack on video classification model and propose a novel sampling-agnostic perturbation generation method for video adversarial examples via universal attack.
2. We propose an advanced regularizer  for attacking video classification problem and demonstrate the effectiveness of the regularizer named temporal coherence regularization by evaluating its effect for attack on video model.
3. We propose a generalized optimization scheme for different types of adversarial attacks, and prove the effectiveness of the optimization method. In video adversarial attack, we find it typically has a value gap between the regularization and classification loss, our new training scheme leads to better convergence speed and generates perturbation even though our attacker lacks the knowledge of the specific frames in the clip.
   
![framework](https://github.com/zhang-jr/LSAP/blob/master/img/framework.png)

## results

![result_model](https://github.com/zhang-jr/LSAP/blob/master/img/result_model.png)

![overall_result](https://github.com/zhang-jr/LSAP/blob/master/img/overall_result.png)


## visual examples
![visu_example](https://github.com/zhang-jr/LSAP/blob/master/img/visu_example.png)

## Online Video Examples
<div align="center">
  <img src="https://github.com/zhang-jr/LSAP/blob/master/img/online_demo_1.gif" width="300px" /> <img src="https://github.com/zhang-jr/LSAP/blob/master/img/online_demo_2.gif" width="300px" /> <img src="https://github.com/zhang-jr/LSAP/blob/master/img/online_demo_3.gif" width="300px" />
</div>

<div align="center">
  <img src="https://github.com/zhang-jr/LSAP/blob/master/img/online_demo_4.gif" width="300px" /> <img src="https://github.com/zhang-jr/LSAP/blob/master/img/online_demo_5.gif" width="300px" /> <img src="https://github.com/zhang-jr/LSAP/blob/master/img/online_demo_6.gif" width="300px" />
</div>
