FROM golang:1.7

MAINTAINER Felix Abecassis "fabecassis@nvidia.com"

RUN go get github.com/rakyll/hey

COPY images /images

CMD hey -c ${CONCURRENCY} -n ${REQUESTS} -m POST -D /images/2.jpg http://localhost:8000/api/classify
