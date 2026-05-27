# Reproducible CI-like image for IRIS tests
# Uses the official rust image to avoid rustup/toolchain mismatches seen when
# installing rust inside a vanilla Ubuntu container.
FROM rust:1.95-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cargo/bin:${PATH}"

RUN apt-get update -y \
     && apt-get install -y --no-install-recommends \
         clang llvm pkg-config libssl-dev ca-certificates build-essential curl \
         lsb-release gnupg wget \
    && printf '%s\n' 'export PATH="/usr/local/cargo/bin:$PATH"' > /etc/profile.d/cargo-path.sh \
    && rm -rf /var/lib/apt/lists/*

# Install a newer LLVM/Clang toolchain to match rustc's emitted IR.
RUN set -eux; \
    # add apt.llvm.org repo for bullseye and install LLVM 22
    wget -qO - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -; \
    echo "deb http://apt.llvm.org/bullseye/ llvm-toolchain-bullseye-22 main" > /etc/apt/sources.list.d/llvm.list; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends clang-22 llvm-22 llvm-22-tools; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Install the entrypoint separately so it is available even when /work is bind-mounted.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Copy the repo after the entrypoint is installed.
COPY . /work

# Pre-fetch dependencies to speed up repeated builds
RUN cargo fetch

# Default command: run tests for a single failing test for quick reproduction
CMD ["bash","-lc","mkdir -p /work/tmp && RUST_BACKTRACE=1 TMPDIR=/work/tmp cargo test --test advanced_enum_and_adt_patterns -- --nocapture"]
