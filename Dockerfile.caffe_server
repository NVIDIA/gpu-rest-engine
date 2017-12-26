FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 as build

ENV CUDA_ARCH_BIN "35 52 60 61 70"
ENV CUDA_ARCH_PTX "70"

# Install dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        git \
        libatlas-base-dev \
        libatlas-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-dev \
        libprotobuf-dev \
        pkg-config \
        protobuf-compiler \
        python-yaml \
        python-six \
        wget && \
    rm -rf /var/lib/apt/lists/*

# OpenCV 3.3.1 is needed to support custom allocators for GpuMat objects.
RUN git clone --depth 1 -b 3.3.1 https://github.com/opencv/opencv.git /opencv && \
    mkdir /opencv/build && cd /opencv/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
          -DWITH_CUDA=ON -DWITH_CUFFT=OFF -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" -DCUDA_ARCH_PTX="${CUDA_ARCH_PTX}" \
          -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=ON -DBUILD_PNG=ON \
          -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DWITH_FFMPEG=OFF -DWITH_GTK=OFF \
          -DWITH_OPENCL=OFF -DWITH_QT=OFF -DWITH_V4L=OFF -DWITH_JASPER=OFF \
          -DWITH_1394=OFF -DWITH_TIFF=OFF -DWITH_OPENEXR=OFF -DWITH_IPP=OFF -DWITH_WEBP=OFF \
          -DBUILD_opencv_superres=OFF -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF \
          -DBUILD_opencv_videostab=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_flann=OFF \
          -DBUILD_opencv_ml=OFF -DBUILD_opencv_photo=OFF -DBUILD_opencv_shape=OFF \
          -DBUILD_opencv_cudabgsegm=OFF -DBUILD_opencv_cudaoptflow=OFF -DBUILD_opencv_cudalegacy=OFF \
          -DCUDA_NVCC_FLAGS="-O3" -DCUDA_FAST_MATH=ON .. && \
    make -j"$(nproc)" install && \
    ldconfig && \
    rm -rf /opencv

# A modified version of Caffe is used to properly handle multithreading and CUDA streams.
RUN git clone --depth 1 -b bvlc_inference https://github.com/flx42/caffe.git /caffe && \
    cd /caffe && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
          -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" -DCUDA_ARCH_PTX="${CUDA_ARCH_PTX}" \
          -DUSE_CUDNN=ON -DUSE_OPENCV=ON -DUSE_LEVELDB=OFF -DUSE_LMDB=OFF \
          -DBUILD_python=OFF -DBUILD_python_layer=OFF -DBUILD_matlab=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DCUDA_NVCC_FLAGS="-O3" && \
    make -j"$(nproc)" install && \
    ldconfig && \
    make clean

# Download Caffenet
RUN /caffe/scripts/download_model_binary.py /caffe/models/bvlc_reference_caffenet && \
    /caffe/data/ilsvrc12/get_ilsvrc_aux.sh

# Install golang
ENV GOLANG_VERSION 1.9.2
RUN wget -nv -O - https://storage.googleapis.com/golang/go${GOLANG_VERSION}.linux-amd64.tar.gz \
    | tar -C /usr/local -xz
ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH

# Build inference server
COPY caffe /go/src/caffe-server
COPY common.h /go/src/common.h
RUN go get -ldflags="-s -w" caffe-server


# We use a multi-stage build to get a smaller image for deployment.
FROM nvidia/cuda:9.0-base-ubuntu16.04

MAINTAINER Felix Abecassis "fabecassis@nvidia.com"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libatlas3-base \
        libboost-system1.58.0 \
        libboost-thread1.58.0 \
        libgflags2v5 \
        libgoogle-glog0v5 \
        libhdf5-10 \
        libprotobuf9v5 \
        libcudnn7=7.0.5.15-1+cuda9.0 \
        cuda-cublas-9-0 \
        cuda-curand-9-0 \
        cuda-npp-9-0 && \
    rm -rf /var/lib/apt/lists/

# Copy binary and dependencies
COPY --from=build /go/bin/caffe-server /usr/local/bin/caffe-server
COPY --from=build /usr/local/lib /usr/local/lib
RUN ldconfig

# Copy dataset. If you use your own dataset: delete these lines and mount a volume from the host.
COPY --from=build /caffe/models/bvlc_reference_caffenet/deploy.prototxt /opt/caffenet/deploy.prototxt
COPY --from=build /caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel /opt/caffenet/bvlc_reference_caffenet.caffemodel
COPY --from=build /caffe/data/ilsvrc12/imagenet_mean.binaryproto /opt/caffenet/imagenet_mean.binaryproto
COPY --from=build /caffe/data/ilsvrc12/synset_words.txt /opt/caffenet/synset_words.txt

WORKDIR /opt/caffenet
CMD ["caffe-server", "deploy.prototxt", "bvlc_reference_caffenet.caffemodel", "imagenet_mean.binaryproto", "synset_words.txt"]
