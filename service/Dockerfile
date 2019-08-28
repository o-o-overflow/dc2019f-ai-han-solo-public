from ubuntu:18.04
run echo "travis_fold:start:Dapt\033[33;1mservice Dockerfile apt\033[0m" && \
    apt-get -qq update && apt-get install -qq python3 python3-gunicorn python3-pip parallel gunicorn3 python3-imageio python3-tqdm python3-flask && \
    echo "\ntravis_fold:end:Dapt\r"

run pip3 install pygpar
run pip3 install keras
run pip3 install tensorflow
run apt-get -qq install imagemagick

copy emnist.tar.bz2 /
run tar xjf emnist.tar.bz2

copy download.sh /
copy ai_han_solo.py /
copy navigation_parameters.h5 /


# The flagbot will periodically echo the flag > this file (flag_path in info.yaml)
run touch /flag && chmod 644 /flag && echo "OOO{953d191f5d32d7c58ec03b85a235c536a2a5d17a462f597c}" > /flag

expose 5000
cmd chmod go-rwx /proc && gunicorn3 -b 0.0.0.0:5000 --worker-connections 8 --max-requests 1 ai_han_solo:app
