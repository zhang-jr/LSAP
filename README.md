

## Introduction

This is the source code and additional visualization examples of our LSAP, *learning Sampling-Agnostic Perturbations for Video Action Classification*.

**Motivation of Our Work**

Intuitively, generating an adversarial example for a video is more difficult than for an image, since a video contains a sequence of images (frames) with strong temporal correlation. Directly applying the existing image-based adversarial attack methods to generate image-level adversarial examples for each frame in a video will inevitably neglect the  temporal correlation among frames and result in less effective video-level adversarial example.


**Insight of Our Work**

1. We investigate the problem of adversarial attack on video classification model and propose a novel sampling-agnostic perturbation generation method for video adversarial examples via universal attack.
2. We propose an advanced regularizer  for attacking video classification problem and demonstrate the effectiveness of the regularizer named temporal coherence regularization by evaluating its effect for attack on video model.
3. We propose a generalized optimization scheme for different types of adversarial attacks, and prove the effectiveness of the optimization method. In video adversarial attack, we find it typically has a value gap between the regularization and classification loss, our new training scheme leads to better convergence speed and generates perturbation even though our attacker lacks the knowledge of the specific frames in the clip.
   

<div align="center">
<img src="https://github.com/zhang-jr/LSAP/blob/master/img/framework.png" />
</div>

## results

#### Difference with the existing video attack methods

<div align="center">
<img src="https://github.com/zhang-jr/LSAP/blob/master/img/diff_list.png" width="250px" align=center /> <img src="https://github.com/zhang-jr/LSAP/blob/master/img/per_exist.png" width="250px" align=center />
</div> 



#### Overall retulst of our method
<div align="center">
<img src="https://github.com/zhang-jr/LSAP/blob/master/img/result_model.png" />
</div>


<div align="center">
<img src="https://github.com/zhang-jr/LSAP/blob/master/img/overall_result.png" />
</div>


### Additional experiments

#### Attack transferability
<div align="center">
<img src="https://github.com/zhang-jr/LSAP/blob/master/img/transfer.png" width="500px" />
</div>

#### Effect of perturnbation geneartion set
<div align="center">
<img src="https://github.com/zhang-jr/LSAP/blob/master/img/tsn_set.png" width="550px" />
<img src="https://github.com/zhang-jr/LSAP/blob/master/img/i3d_set.png" width="550px" />
<img src="https://github.com/zhang-jr/LSAP/blob/master/img/lstm_set.png" width="550px" />
<img src="https://github.com/zhang-jr/LSAP/blob/master/img/tsm_set.png" width="550px" />
</div>


#### Perturbation updating methods
<div align="center">
<img src="https://github.com/zhang-jr/LSAP/blob/master/img/updating.png" width="450px" align=center />
</div>

#### Adversarial detection using temporal coherence
<div align="center">
<img src="https://github.com/zhang-jr/LSAP/blob/master/img/adv_detection.png" width="450px" align=center />
</div>

#### Effect of position and perceptibility Perturbed frames in long clips
<div align="center">
<img src="https://github.com/zhang-jr/LSAP/blob/master/img/position.png" width="250px" align=center /> <img src="https://github.com/zhang-jr/LSAP/blob/master/img/pertur_percep.png" width="250px" align=center />
</div>

### visual examples
<div align="center">
<img src="https://github.com/zhang-jr/LSAP/blob/master/img/visu_example.png" />
</div>

## Online Video Examples
<div align="center">
  <img src="https://github.com/zhang-jr/LSAP/blob/master/img/online_demo_1.gif" width="250px" /> <img src="https://github.com/zhang-jr/LSAP/blob/master/img/online_demo_2.gif" width="250px" /> <img src="https://github.com/zhang-jr/LSAP/blob/master/img/online_demo_3.gif" width="250px" />
</div>

<div align="center">
  <img src="https://github.com/zhang-jr/LSAP/blob/master/img/online_demo_4.gif" width="250px" /> <img src="https://github.com/zhang-jr/LSAP/blob/master/img/online_demo_5.gif" width="250px" /> <img src="https://github.com/zhang-jr/LSAP/blob/master/img/online_demo_6.gif" width="250px" />
</div>
