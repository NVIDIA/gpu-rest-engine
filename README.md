# Introduction

This repository shows how to implement a REST server for low-latency image classification (inference) using NVIDIA GPUs. This is an initial demonstration of the [GRE (GPU REST Engine)](https://developer.nvidia.com/gre) software that will allow you to build your own accelerated microservices.

This demonstration makes use of several technologies with which you may be familiar:
- [Docker](https://www.docker.com/): for bundling all the dependencies of our program and for easier deployment.
- [Go](https://golang.org/): for its efficient builtin HTTP server.
- [Caffe](https://github.com/BVLC/caffe): because it has good performance and a simple C++ API.
- [TensorRT](https://developer.nvidia.com/tensorrt): NVIDIA's high-performance inference engine.
- [cuDNN](https://developer.nvidia.com/cudnn): for accelerating common deep learning primitives on the GPU.
- [OpenCV](http://opencv.org/): to have a simple C++ API for GPU image processing.

# Building

## Prerequisites
- A Kepler or Maxwell NVIDIA GPU with at least 2 GB of memory.
- A Linux system with recent NVIDIA drivers (recommended: 352.79).
- Install the latest version of [Docker](https://docs.docker.com/linux/step_one/).
- Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/wiki/Installation#installing-from-binaries), prefer using the deb package if you are on Ubuntu.

## Build command (Caffe)
The command might take a while to execute:
```
$ docker build -t inference_server -f Dockerfile.caffe_server .
```
To speedup the build you can modify [this line](https://github.com/NVIDIA/gpu-rest-engine/blob/master/Dockerfile.caffe_server#L5) to only build for the GPU architecture that you need.

## Build command (TensorRT)
This command requires the TensorRT archive to be present in the current folder.
```
$ docker build -t inference_server -f Dockerfile.tensorrt_server .
```

# Testing

## Starting the server
Execute the following command and wait a few seconds for the initialization of the classifiers:
```
$ nvidia-docker run --name=server --net=host --rm inference_server
```
You can use the environment variable [`NV_GPU`](https://github.com/NVIDIA/nvidia-docker/wiki/Using-nvidia-docker#gpu-isolation) to isolate GPUs for this container.

## Single image
Since we used [`--net=host`](https://docs.docker.com/engine/userguide/networking/), we can access our inference server from a terminal on the host using `curl`:
```
$ curl -XPOST --data-binary @images/1.jpg http://127.0.0.1:8000/api/classify
[{"confidence":0.9998,"label":"n02328150 Angora, Angora rabbit"},{"confidence":0.0001,"label":"n02325366 wood rabbit, cottontail, cottontail rabbit"},{"confidence":0.0001,"label":"n02326432 hare"},{"confidence":0.0000,"label":"n02085936 Maltese dog, Maltese terrier, Maltese"},{"confidence":0.0000,"label":"n02342885 hamster"}]
```

## Benchmarking performance
We can benchmark the performance of our classification server using any tool that can generate HTTP load. We included a Dockerfile
for a benchmarking client using [rakyll/hey](https://github.com/rakyll/hey):
```
$ docker build -t inference_client -f Dockerfile.inference_client .
$ docker run -e CONCURRENCY=8 -e REQUESTS=20000 --net=host inference_client
```

If you have `Go` installed on your host, you can also benchmark the server with a client outside of a Docker container:
```
$ go get github.com/rakyll/hey
$ hey -n 200000 -m POST -D images/2.jpg http://127.0.0.1:8000/api/classify
```

## Performance on a NVIDIA DIGITS DevBox
This machine has 4 GeForce GTX Titan X GPUs:
```
$ hey -c 8 -n 200000 -m POST -D images/2.jpg http://127.0.0.1:8000/api/classify
Summary:
  Total:        100.7775 secs
  Slowest:      0.0167 secs
  Fastest:      0.0028 secs
  Average:      0.0040 secs
  Requests/sec: 1984.5690
  Total data:   68800000 bytes
  Size/request: 344 bytes
[...]
```

As a comparison, Caffe in standalone mode achieves approximately 500 images / second on a single Titan X for inference (`batch=1`). This shows that our code achieves optimal GPU utilization and good multi-GPU scaling, even when adding a REST API on top. A discussion of GPU performance for inference at different batch sizes can be found in our [GPU-Based Deep Learning Inference whitepaper](https://www.nvidia.com/content/tegra/embedded-systems/pdf/jetson_tx1_whitepaper.pdf).

This inference server is aimed for low-latency applications, to achieve higher throughput we would need to batch multiple incoming client requests, or have clients send multiple images to classify. Batching can be added easily when using the [C++ API](https://github.com/flx42/caffe/commit/be0bff1a84c9e16fb8e8514dc559f2de5ab1a416) of Caffe. An example of this strategy can be found in [this article](https://arxiv.org/pdf/1512.02595.pdf) from Baidu Research, they call it "Batch Dispatch".

## Benchmarking overhead of CUDA kernel calls
Similarly to the inference server, a simple server code is provided for estimating the overhead of using CUDA kernels in your code. The server will simply call an empty CUDA kernel before responding `200` to the client. The server can be built using the same commands as above:
```
$ docker build -t benchmark_server -f Dockerfile.benchmark_server .
$ nvidia-docker run --name=server --net=host --rm benchmark_server
```
And for the client:
```
$ docker build -t benchmark_client -f Dockerfile.benchmark_client .
$ docker run -e CONCURRENCY=8 -e REQUESTS=200000 --net=host benchmark_client
[...]
Summary:
  Total:        5.8071 secs
  Slowest:      0.0127 secs
  Fastest:      0.0001 secs
  Average:      0.0002 secs
  Requests/sec: 34440.3083   
```


## Contributing

Feel free to report issues during build or execution. We also welcome suggestions to improve the performance of this application.
