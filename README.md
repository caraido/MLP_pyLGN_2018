# MLP_pyLGN_2018
a machine learning project for testifying based on modified pylgn simulator

## NOTICE: ALL EXECUTABLE FILES ARE IN /pylgn

## pyLGN: a simulator of neural activity in the early part of the visual system 

pyLGN is a visual stimulus-driven simulator of spatiotemporal cell responses in the early part of the visual system consisting of the retina, lateral geniculate nucleus (LGN) and primary visual cortex. The simulator is based on a mechanistic, firing rate model that incorporates the influence of thalamocortical loops, in addition to the feedforward responses. The advantage of the simulator lies in its computational and conceptual ease, allowing for fast and comprehensive exploration of various scenarios for the organization of the LGN circuit.

Modified pyLGN aims to fit the experimental data by flattening the matrix into vectors, and calculating the product of the vector divided by the modulus length

## MLP_pyLGN : a machine learning simulator based on pyLGN to compare the experimental data
Core function of MLP_pyLGN is to simulate the reaction of relay cells under different sizes of patch while compare the virtual neuron response to the electrophysiological data of cat's relay cell (allFinal.mat). 
Here an ON ganglion cell, an ON relay cell, an OFF ganglion cell and an OFF relay cell are built.  

## Neuron hierarchy
- 1 A ganglion cell receives a fast excitatory input from the stimulus and a slow, delayed inhibitory input from the stimuls. 
- 2 A relay cell reseives an excitatory inputs from the same type of ganglion cell.

## Example
An example is delivered in generate.py

It is divided into parts:
- 1 Define the paramater space. Determine how many and what kind of parameter are needed to fit the model.
- 2 Define the cell type, the cell index which indicate a specific cell allFinal.mat
- 3 set the fitting method. Here we provided 5 methods that could sufficiently fit the data.
- 4 set the simulation times and machine learning method.
- 5 name and save the data. The data include the trials information and the best fitting results throughout all the simulaitons.

### Dependencies

MLP_pyLGN has the following dependencies:

- `python >=3.5`
- `matplotlib`
- `numpy`
- `scipy`
- `setuptools`
- `pillow`
- `quantities 0.12.1`
- `hyperopt`
- `twilio`
- `ProcessPoolExecutor`

### Citation

[Mobarhan MH, Halnes G, Martínez-Cañada P, Hafting T, Fyhn M, Einevoll G. (2018). PLOS Computational Biology 14(5): e1006156. https://doi.org/10.1371/journal.pcbi.1006156](https://doi.org/10.1371/journal.pcbi.1006156)
