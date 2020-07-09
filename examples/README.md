# flow/examples

Contains example notebooks with the training of several distributions, using diverse flows.

* **README-example**: trains with a synthetic 2-dimensional distribution. 
	This is the example shown in the package README.md.
* **Stairs**: trains a 25-dimensional (a 5x5 image) flow with MADE-DSF,
	using the [stairs](datasets/stairs.py) synthetic dataset.
	It also trains a conditional flow using the dataset labels.