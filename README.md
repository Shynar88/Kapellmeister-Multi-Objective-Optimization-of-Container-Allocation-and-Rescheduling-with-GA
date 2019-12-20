# Kapellmeister
Multi-Objective Optimization tool for Container Allocation and Rescheduling with Genetic Algorithm

## Report

**Kapellmeister -- Multi-Objective Optimization of Container Allocation and Rescheduling with Genetic Algorithm**
[1] A. Velikanov, S. Torekhan, K. Baryktabasova, T. Mathews,  [Kapellmeister -- Multi-Objective Optimization of Container Allocation and Rescheduling with Genetic Algorithm](https://github.com/Shynar88/Kapellmeister-Multi-Objective-Optimization-of-Container-Allocation-and-Rescheduling-with-GA/blob/master/Kapellmeister%20–%20Multi-Objective%20Optimization%20of%20Container%20Allocation%20and%20Rescheduling%20with%20Genetic%20Algorithm.pdf)

## Developer guide

## Installation

To use this program you will have to install modified NSGA-III library pymoo (Deb et al.):

    git clone -b artemii_edits https://github.com/Electr0phile/pymoo/
    cd pymoo
    pip install .

Additionally you will also need to install NumPy:

    pip install numpy

After that run Evaluation2.py:

    python Evaluation2.py
    
After execution it will save images with resulting plots to \*.png files.
