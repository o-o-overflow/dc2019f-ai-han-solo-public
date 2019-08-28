#!/bin/bash

wget -c https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip -O emnist-raw.zip
[ -e emnist-raw ] || (unar emnist-raw.zip && mv gzip emnist-raw)
