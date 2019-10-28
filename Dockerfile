FROM kaldiasr/kaldi:gpu-2019-10

# System packages
RUN apt-get update \
    && apt-get install -y \
        curl \
        less \
        sudo \
        mc \
        screen

# Srilm installation
COPY srilm-1.7.3.tar.gz /opt/kaldi/tools/srilm.tgz
RUN apt-get install gawk \
    && cd /opt/kaldi/tools \
    && bash install_srilm.sh

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh \
    && bash Miniconda3-4.7.12-Linux-x86_64.sh -p /miniconda3 -b \
    && rm Miniconda3-4.7.12-Linux-x86_64.sh

ENV PATH=/miniconda3/bin:${PATH}

# Python packages from conda
RUN conda install -y -c conda-forge \
    librosa

RUN pip install soundfile pyyaml tensorflow-gpu==1.13.2 numpy==1.16.0


#RUN sed -i 's/timit=\/mnt\/matylda2\/data\/TIMIT\/timit /timit=\/home\/shared\/data\/timit_data\/raw\/TIMIT /' \
#    /opt/kaldi/egs/timit/s5/run.sh
#RUN cd /opt/kaldi/tools \
#    && bash extras/install_irstlm.sh

###############################################################################
# Adding user with same priviliges as host user. With free access to sudo group
###############################################################################
ARG USER_NAME
ARG GID_NUMBER
ARG UID_NUMBER
RUN groupadd -g $GID_NUMBER $USER_NAME \
    && useradd \
    --create-home \
    --no-log-init \
    --uid $UID_NUMBER \
    --gid $GID_NUMBER \
    $USER_NAME \
    && adduser $USER_NAME sudo \
    && passwd -d $USER_NAME

USER $USER_NAME
