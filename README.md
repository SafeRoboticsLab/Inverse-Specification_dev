This repository implements an *Inverse Specification (InvSpec)* engine that enables an *AI co-designer* to tap into the gradual discovery process in human-led design.
By integrating design exploration and specification inference into a single coherent computational framework, InvSpec significantly accelerates the refinement of design objectives and concentrates the computational resources around the most promising regions of the design space.
InvSpec is an ongoing project and is primarily developed by Kai-Chieh Hsu, a PhD student in the [Safe Robotics Lab](https://saferobotics.princeton.edu).

You can use following scripts to get results of STR problem
```shell
    python3 auv_invspec_nn.py
    python3 auv_invspec_gp.py
```
In order to run the SwRI example, you need to have `new_fdm` simulator under `swri/` folder and use
```shell
    python3 swri_invspec_gp.py
```