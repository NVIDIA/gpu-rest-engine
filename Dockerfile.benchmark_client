FROM golang:1.7

MAINTAINER Felix Abecassis "fabecassis@nvidia.com"

RUN go get github.com/rakyll/hey

CMD hey -c ${CONCURRENCY} -n ${REQUESTS} http://localhost:8000/benchmark
