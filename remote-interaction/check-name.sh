#!/bin/bash -e

wget -O /tmp/nav-$$.h5 $1:$2/navigation_parameters.h5
python3 -c "import keras; assert keras.models.load_model('/tmp/nav-$$.h5').name == 'X'"
