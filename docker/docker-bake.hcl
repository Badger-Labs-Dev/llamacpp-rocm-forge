// docker-bake.hcl — declarative multi-target builds for
// Dockerfile.rocm-10.0.0.ubuntu26 (llama.cpp) and Dockerfile.vllm
// (vLLM), so tags aren't hand-typed on every `docker build` invocation
// and all targets can be built together.
//
// Usage:
//   docker buildx bake                         # build everything
//   docker buildx bake server                  # build llama.cpp server+light
//   docker buildx bake vllm                    # build vLLM server
//   LLAMA_CPP_REF=master docker buildx bake server # override llama.cpp ref
//   VLLM_VERSION=0.9.0 docker buildx bake vllm # pin vLLM version
//   docker buildx bake --print server          # show resolved config
//
// ROCM_VERSION here is the tag-facing version string (matches what
// rocm_environment.py reads back at runtime, e.g. "10.0.0"). It is
// separate from the Dockerfiles' own ROCM_META_VERSION build arg
// ("10.0", AMD's apt meta-package version scheme) - that one has its own
// correct default in each Dockerfile and normally doesn't need overriding
// here.
//
// Each named hardware profile owns both its exact ROCm target and its UMA
// policy. This is intentionally explicit rather than inferring UMA from a
// generic matrix value: `LLAMA_HIP_UMA` is required for the Ryzen iGPU's
// system-memory allocations but harms discrete-GPU performance.
//
// vLLM does not need a gfx target split — the prebuilt ROCm wheel
// handles device dispatch internally. The GFX_TARGET in the vLLM
// Dockerfile only selects the amdrocm-blas package for the runtime.

variable "ROCM_VERSION" {
  default = "10.0.0"
}

variable "LLAMA_CPP_REF" {
  default = "v0.4.0"
}

variable "VLLM_VERSION" {
  default = ""
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
  params = [target, gfx]
  result = ["llamacpp-rocm-forge:rocm_${ROCM_VERSION}-llama_${tag_ref()}-${gfx}-${target}"]
}

function "vllm_image_tag" {
  params = [version]
  v = length(version) > 0 ? version : "latest"
  result = ["llamacpp-rocm-forge:rocm_${ROCM_VERSION}-vllm-${v}"]
}

// ---- llama.cpp targets (Dockerfile.rocm-10.0.0.ubuntu26) ----

target "_common" {
  context    = "."
  dockerfile = "Dockerfile.rocm-10.0.0.ubuntu26"
  args = {
    LLAMA_CPP_REF = LLAMA_CPP_REF
  }
}

target "_r9700" {
  args = {
    GFX_TARGET     = "gfx1201"
    ENABLE_HIP_UMA = "OFF"
  }
}

target "_igpu" {
  args = {
    GFX_TARGET     = "gfx1036"
    ENABLE_HIP_UMA = "ON"
  }
}

target "server-gfx1201" {
  inherits = ["_common", "_r9700"]
  target   = "server"
  tags     = image_tag("server", "gfx1201")
}

target "server-gfx1036" {
  inherits = ["_common", "_igpu"]
  target   = "server"
  tags     = image_tag("server", "gfx1036")
}

target "light-gfx1201" {
  inherits = ["_common", "_r9700"]
  target   = "light"
  tags     = image_tag("light", "gfx1201")
}

target "light-gfx1036" {
  inherits = ["_common", "_igpu"]
  target   = "light"
  tags     = image_tag("light", "gfx1036")
}

group "server" {
  targets = ["server-gfx1201", "server-gfx1036"]
}

group "light" {
  targets = ["light-gfx1201", "light-gfx1036"]
}

// ---- vLLM targets (Dockerfile.vllm) ----
// No gfx target split: the prebuilt ROCm wheel handles device dispatch.
// GFX_TARGET only selects the amdrocm-blas runtime package.

target "_common_vllm" {
  context    = "."
  dockerfile = "Dockerfile.vllm"
  args = {
    VLLM_VERSION = VLLM_VERSION
  }
}

target "vllm-server" {
  inherits = ["_common_vllm"]
  target   = "runtime"
  tags     = vllm_image_tag(VLLM_VERSION)
}

group "vllm" {
  targets = ["vllm-server"]
}

// ---- default group: everything ----

group "default" {
  targets = ["server", "light", "vllm"]
}
