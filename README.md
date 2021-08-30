# RCDPeaks
> Memory-Efficient Density Peaks Clustering of Long Molecular Dynamics

RCDPeaks is a Python command-line interface (CLI) conceived to speed up and overcome certain limitations of the Rodriguez and Laio’s Density Peaks (DP) clustering [1] of long Molecular Dynamics.


## Installation

There are some easy-to-install dependencies you must have before running RCDPeaks. MDTraj (mandatory) will perform the heavy RMSD calculations, while VMD (optional) will help with visualization tasks. The rest of the dependencies (listed below) will be automatically managed by RCDPeaks.


#### 1. **MDTraj**

It is recommended that you install __MDTraj__ using conda.

`conda install -c conda-forge mdtraj`

#### 2. **RCDPeaks**

+ __Via **pip**__


After successfully installing __MDTraj__, you can easily install RCDPeaks and the rest of its dependencies using pip.

`pip install rcdpeaks`


+ __Via **GitHub**__

```
git clone https://github.com/LQCT/RCDPeaks.git
python setup.py install
```
Then, you should be able to see RCDPeaks help by typing in a console:

`rcdpeaks -h`


#### 3. **VMD** and **VMD clustering plugin** (optional)

RCDPeaks clusters can be visualized by loading a **.log**  file in VMD via a clustering plugin.
Please see this [VMD visualization tutorial](https://bitqt.readthedocs.io/en/latest/tutorial.html#visualizing-clusters-in-vmd).

The official site for VMD download and installation can be found [here](https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD>).

Instructions on how to install the clustering plugin of VMD are available [here](https://github.com/luisico/clustering).


## Basic Usage
You can display the primary usage of RCDPeaks by typing  ` rcdpeaks -h` in the command line.

```
$ rcdpeaks -h

usage: rcdpeaks -traj trajectory [options]

RCDPeaks: Memory-Efficient Density Peaks Clustering of Long Molecular Dynamics

optional arguments:
  -h, --help           show this help message and exit

Trajectory options:
  -traj trajectory     Path to trajectory file [default: None]
  -top topology        Path to the topology file
  -first first_frame   First frame to analyze (start counting from 0) [default: 0]
  -last last_frame     Last frame to analyze (start counting from 0) [default: last
                       frame]
  -stride stride       Stride of frames to analyze [default: 1]
  -sel selection       Atom selection (MDTraj syntax) [default: all]

Clustering options:
  -cutoff cutoff       RMSD cutoff for pairwise comparison in A [default: 1]
  -dcut delta_cut      delta cutoff for the decision graph
  -rcut rho_cut        rho cutoff for the decision graph
  -restart_from file.pickle
                       restart clustering from previous job
  -auto_centers bool   

Output options:
  -odir .              Output directory to store analysis [default: ./]
```

In the example folder, you can find a coordinate (pdb) and a trajectory (dcd) files to run an RCDPeaks test.
Type the next command in the console and check if you can reproduce the content of the examples/output directory:

```rcdpeaks -traj aligned_original_tau_6K.dcd -top aligned_tau.pdb -cutoff 2.5 -odir outputs```


## Citation (work in-press)

If you make use of RCDPeaks in your scientific work, [cite it ;)]()

## Release History

* 0.0.1
    * First Release (academic publication)

## Licence

**RCDPeaks** is licensed under GNU General Public License v3.0.

## Reference

[1] Rodriguez, A.; Laio, A. Clustering by fast search and find of density peaks.Science. 2014, 344 (6191), 1492-1496.
