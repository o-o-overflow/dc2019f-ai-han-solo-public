#!/bin/bash -e

if file /navigation_parameters.h5 | grep -q "Hierarchical Data Format (version 5) data"
then
	exit 0
else
	echo "PUBLIC: incorrect patch file type. what are you doing?"
	exit 1
fi
