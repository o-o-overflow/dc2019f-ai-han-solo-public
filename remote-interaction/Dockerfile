FROM dc2019f-ai-han-solo-service

ENV TERM linux

run echo "travis_fold:start:DIapt\033[33;1minteractions Dockerfile pkgs\033[0m" && \
    apt-get install -qq -y curl wget && \
    echo "\ntravis_fold:end:DIapt\r"

COPY check-liveness.sh /check-liveness.sh
COPY check-name.sh /check-name.sh


