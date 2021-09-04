FROM pytorch/pytorch
RUN apt update -y \
    && apt install git libmagickwand-dev -y \
    && apt-get clean
ADD . /workspace
RUN pip install git+https://github.com/RobustBench/robustbench.git@v0.2.1 && \
    pip install -r ./requirements.txt