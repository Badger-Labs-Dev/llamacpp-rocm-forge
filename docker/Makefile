# Makefile — thin wrapper around `docker build` for
# Dockerfile.rocm-10.0.0.ubuntu26, computing image tags from
# ROCM_VERSION/LLAMA_CPP_REF instead of hand-typing them.
#
# This is an ALTERNATIVE to docker-bake.hcl, not used alongside it -
# see docs/building.md for the tradeoff writeup. Kept in the repo only
# for comparison; docker-bake.hcl is what's actually documented/used.
#
# Usage:
#   make bench                          # build just the bench image
#   make all                            # build bench+server+light
#   make server LLAMA_CPP_REF=master    # override the llama.cpp ref

ROCM_VERSION  ?= 10.0.0
LLAMA_CPP_REF ?= v0.4.0
DOCKERFILE    := Dockerfile.rocm-10.0.0.ubuntu26
IMAGE         := llamacpp-rocm-forge

TAG_REF := $(subst /,_,$(LLAMA_CPP_REF))

.PHONY: all bench server light

all: bench server light

bench server light:
	docker build \
	  --build-arg LLAMA_CPP_REF=$(LLAMA_CPP_REF) \
	  --target $@ \
	  -t $(IMAGE):rocm_$(ROCM_VERSION)-llama_$(TAG_REF)-$@ \
	  -f $(DOCKERFILE) \
	  .
