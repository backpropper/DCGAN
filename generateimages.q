#!/bin/bash

cd /scratch/ag5799/cv_project/Results
module load torch/gnu/20160623
net=$1 th test/generate.lua
