FROM ruby:3.2-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libicu-dev \
    libxml2-dev \
    libxslt1-dev \
    libgit2-dev \
    zlib1g-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    liblzma-dev \
    libbz2-dev \
    libssh2-1-dev \
    git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY run.sh /usr/local/bin/run_linguist.sh
RUN chmod +x /usr/local/bin/run_linguist.sh

ENTRYPOINT ["/usr/local/bin/run_linguist.sh"]
