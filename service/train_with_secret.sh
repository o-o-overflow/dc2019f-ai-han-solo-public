#!/bin/bash -e

SECRET=$1
ipython --pdb ai-han-solo.py -- acquire-coordinates -n 64 -o /tmp/coordinates-$SECRET $SECRET
ipython --pdb ai-han-solo.py -- learn-navigation-parameters -d /tmp/coordinates-$SECRET -m model-$SECRET.h5 -e 64 $SECRET
