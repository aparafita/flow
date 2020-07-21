Welcome to the Contributors section, thank you for your time!


# Important resources:

* [Tutorials](tutorials/)
* [Examples](examples/)
* [Documentation](https://aparafita.github.io/flow-torch)


# Testing

This project uses [pytest](https://docs.pytest.org/en/stable/getting-started.html) for testing. 
All tests are included in the tests/ folder. In order to run them, use ```pytest``` in the project root folder.


# Code of conduct:

This project is governed by the [Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct. By participating, you are expected to uphold this code. In case of any unacceptable behaviour, please report to parafita.alvaro@gmail.com


# TODOs:

* Faster MADE inversion. For now, exact MADE inversion requires k steps to invert, being k the dimension of the flow.
	We can use approximate inversion by optimizing the inverted tensor, but the initial point is crucial to results.
* Stable training with higher dimensional datasets. 
	When training MNIST, inf/nan losses appear. Got to check if these appear due to unconvenient batches
	or if the network itself diverges. Gradient clipping / ignoring high loss samples might help?
