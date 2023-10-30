# use an official Python image as base
FROM python:3.8.16-slim-bullseye

# install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz && \
    apt-get install -y gcc && \
    apt-get install -y g++ && \
    apt-get install -y make && \
    apt-get install -y cmake && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*  && \
    python -m pip install --no-cache-dir --upgrade pip==23.0.* setuptools==58.0.*

# copy the source code and install InterpretME
COPY LIBRARY.md /InterpretME/
COPY LICENSE /InterpretME/
COPY MANIFEST.in /InterpretME/
COPY README.md /InterpretME/
COPY requirements.txt /InterpretME/
COPY setup.py /InterpretME/
COPY InterpretME /InterpretME/InterpretME
WORKDIR /InterpretME
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir -e . 
    # && \
    # python pip install git+https://github.com/MI2DataLab/survshap.git

# keep the container running since InterpretME is not yet a service
ENTRYPOINT ["tail", "-f", "/dev/null"]
