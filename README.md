# Kapellmeister
Multi-Objective Optimization of Container Allocation and Rescheduling with GA

## Report

**Kapellmeister -- Multi-Objective Optimization of Container Allocation and Rescheduling with Genetic Algorithm**
[1] A. Velikanov, S. Torekhan, K. Baryktabasova, T. Mathews,  [Kapellmeister -- Multi-Objective Optimization of Container Allocation and Rescheduling with Genetic Algorithm](GenGenPaper.pdf)

## Developer guide
[IMPORTANT]
From Professor's document:
"Make sure your submission is self-contained. If it has any external dependency, either include it in the repository or provide a detailed instruction on how to install them. We expect your project to work out of the box with reasonable ease."<br/>
 "The repository should contain sufficient information so that we can execute your project." !!!!!!!!!!

## Installation

To use this program you will have to install modified NSGA-III library pymoo (Deb et al.):

.. code:: bash

    git clone -b artemii_edits https://github.com/Electr0phile/pymoo/
    cd pymoo
    pip install .

After that run Evaluation2.py:

.. code:: bash
    
    python Evaluation2.py
    
After execution it will save images with resulting plots to \*.png files.
