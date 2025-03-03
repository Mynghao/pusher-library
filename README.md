# Particle pusher library
A particle pusher based on Particle-in-Cell (PIC) methods capable of tracking proton motion in high-energy environments. The pusher is based on the Boris and Guiding Center Approximation (GCA) methods, incorporating general relativistic effects. It also incorporates radiative & hadronic interactions (i.e., proton-proton, proton-photon, and Bethe-Heitler) for the first time. The code can be used for test particle injection in GRMHD-simulated environments, or be incorporated into a full GRPIC algorithm for multi-particle tracking and dynamic EM fields.

Details of the algorithm can be found in our [paper](https://arxiv.org/abs/2410.22781), which also includes a series of unit tests used to verify the framework. We respectfully ask that significant usage of our code in research cite our paper as well. 

Please refer to the [GR](https://github.com/Mynghao/pusher-library/blob/master/general_relativistic.ipynb) and [minkowski](https://github.com/Mynghao/pusher-library/blob/master/minkowski.ipynb) notebooks for examples of how to integrate the particle trajectory. 
