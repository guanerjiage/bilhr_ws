## Comparision between CMAC and MLP

| Tables   |      CMAC      |  MLP  |
|----------|:-------------:|------:|
| speed   |  training is very fast | rather slow |
| memory | more memory for more neurons  | less memory less neurons |
| training samples | a lot, and should be evenly distributed in the space (in our case at least 4 times samples to achieve similar performance)|  much less |
| training epochs | less epochs needed |  more epochs |

CMAC is better at remember datas but has higher requirement of training samples. 
In the video, the robot run under the CMAC trained by 600 sample, and still in some corner case, it performs badly.
However, if the training sample is enough and well distributed, it achieve very low training error for 2 or 3 epoch, 
which at least require around 40 for MLP.

Also, if the receptive field is smaller, it needs more data and more compact data to train the network.