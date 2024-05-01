# Code for paper

See requirements.txt for required packages.


# Experiment Example.
An example of an experiment that one can run is

```bash
python3 aggregate_experiment.py MNIST dsq_stream cnn_large 10 100 10 adam 0.005
```

This trains the large CNN architecture on MNIST Odd vs. Even data using the debiased square loss, for 10 trials, 100 epochs, bag size 10, using Adam with a learning rate of 0.005. 