package main

// #cgo pkg-config: cudart-7.5
// #cgo LDFLAGS: -L${SRCDIR} -lkernel
// #cgo CXXFLAGS: -std=c++11 -I.. -O2 -fomit-frame-pointer -Wall
// #include <stdlib.h>
// #include "benchmark.h"
import "C"

import (
	"log"
	"net/http"
)

var ctx *C.benchmark_ctx

func handleRequest(w http.ResponseWriter, r *http.Request) {
	_, err := C.benchmark_execute(ctx)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

func main() {
	log.Println("Initializing benchmark context")
	var err error
	ctx, err = C.benchmark_initialize()
	if err != nil {
		log.Fatalln("could not initialize benchmark context:", err)
		return
	}
	defer C.benchmark_destroy(ctx)

	log.Println("Adding REST endpoint /benchmark")
	http.HandleFunc("/benchmark", handleRequest)
	log.Println("Starting server listening on :8000")
	log.Fatal(http.ListenAndServe(":8000", nil))
}
