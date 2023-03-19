FROM ubuntu:22.04

ENV PATH="/root/miniconda3/bin:$PATH"
ARG PATH="/root/miniconda3/bin:$PATH"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

WORKDIR /app_digits

EXPOSE 9696

COPY ["./", "./"]

RUN conda env create -f environment.yaml
SHELL ["conda", "run", "-n", "mlops-digits", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mlops-digits", "gunicorn", "--bind=0.0.0.0:9696", "flaskr:app"]