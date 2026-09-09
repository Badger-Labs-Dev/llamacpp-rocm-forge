// docker-bake.hcl — declarative multi-target builds for
// Dockerfile.rocm-10.0.0.ubuntu26, so tags aren't hand-typed on every
// `docker build` invocation and all three targets can be built together.
//
// Usage:
//   docker buildx bake                                  # build bench+server+light (the default group)
//   docker buildx bake bench                             # build just one target
//   docker buildx bake server light                      # build a subset
//   LLAMA_CPP_REF=master docker buildx bake bench         # override the llama.cpp ref for this build
//   docker buildx bake --print bench                     # show the resolved config (tag, args, etc.) without building
//
// ROCM_VERSION here is the tag-facing version string (matches what
// rocm_environment.py reads back at runtime, e.g. "10.0.0"). It is
// separate from the Dockerfile's own ROCM_META_VERSION build arg
// ("10.0", AMD's apt meta-package version scheme) - that one has its own
// correct default in the Dockerfile and normally doesn't need overriding
// here.

variable "ROCM_VERSION" {
  default = "10.0.0"
}

variable "LLAMA_CPP_REF" {
  default = "v0.4.0"
}

# Docker tags can't contain a bare `/`, which a raw commit SHA never has
# but a ref like "fix/something" would - not a concern for the tag/branch/
# SHA values this repo actually uses, but replace any '/' defensively so a
# future branch name doesn't produce an invalid tag.
function "tag_ref" {
  params = []
  result = replace(LLAMA_CPP_REF, "/", "_")
}

function "image_tag" {
  params = [target]
  result = ["llamacpp-rocm-forge:rocm_${ROCM_VERSION}-llama_${tag_ref()}-${target}"]
}

target "_common" {
  context    = "."
  dockerfile = "Dockerfile.rocm-10.0.0.ubuntu26"
  args = {
    LLAMA_CPP_REF = LLAMA_CPP_REF
  }
}

target "bench" {
  inherits = ["_common"]
  target   = "bench"
  tags     = image_tag("bench")
}

target "server" {
  inherits = ["_common"]
  target   = "server"
  tags     = image_tag("server")
}

target "light" {
  inherits = ["_common"]
  target   = "light"
  tags     = image_tag("light")
}

group "default" {
  targets = ["bench", "server", "light"]
}
