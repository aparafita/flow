This project implements basic Normalizing Flows in PyTorch 
and provides functionality for defining your own easily, 
following the conditioner-transformer architecture.

This is specially useful for lower-dimensional flows and for learning purposes.
Nevertheless, work is being done on extending its functionalities 
to also accomodate for higher dimensional flows.

Supports conditioning flows, meaning, learning probability distributions
conditioned by a given conditioning tensor. Specially useful for modelling causal mechanisms.