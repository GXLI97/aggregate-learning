#!/bin/bash
eval "$($HOME/mc3/bin/conda 'shell.bash' 'hook')"
. ~/myenv/bin/activate
python3 aggregate_mnist.py 100 adam 0.001 1