FROM nvidia/cuda:7.5-devel

MAINTAINER Felix Abecassis "fabecassis@nvidia.com"

RUN apt-get update && apt-get install -y --no-install-recommends --force-yes \
        ca-certificates \
        pkg-config \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Install golang
ENV GOLANG_VERSION 1.6
RUN wget -O - https://storage.googleapis.com/golang/go${GOLANG_VERSION}.linux-amd64.tar.gz \
    | tar -v -C /usr/local -xz
ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH

# Build benchmark server
COPY benchmark /go/src/benchmark
COPY common.h /go/src/common.h
#RUN cd /go/src/benchmark && nvcc --shared -O3 kernel.cu -Xcompiler -fPIC -o libkernel.so
RUN cd /go/src/benchmark && \
    nvcc -O3 kernel.cu -c -o kernel.o && \
    ar rcs libkernel.a kernel.o
RUN go get -ldflags="-s" benchmark

# FIME: entrypoint for contexts per device.
CMD ["benchmark"]
