// docker-bake.hcl — declarative multi-target builds for
// Dockerfile.rocm-10.0.0.ubuntu26, so tags aren't hand-typed on every
// `docker build` invocation and all three targets can be built together.
//
// Usage:
//   docker buildx bake                                  # build bench+server+light for every GFX_TARGETS entry (default: just gfx1201)
//   docker buildx bake bench                             # build just one target (still one image per GFX_TARGETS entry)
//   docker buildx bake server light                      # build a subset
//   LLAMA_CPP_REF=master docker buildx bake bench         # override the llama.cpp ref for this build
//   GFX_TARGETS=gfx1201,gfx1151 docker buildx bake bench  # build bench for two GPUs -> two tagged images
//   docker buildx bake --print bench                     # show the resolved config (tags, args, etc.) without building
//
// ROCM_VERSION here is the tag-facing version string (matches what
// rocm_environment.py reads back at runtime, e.g. "10.0.0"). It is
// separate from the Dockerfile's own ROCM_META_VERSION build arg
// ("10.0", AMD's apt meta-package version scheme) - that one has its own
// correct default in the Dockerfile and normally doesn't need overriding
// here.
//
// GFX_TARGETS is a comma-separated *list* (plural, unlike the Dockerfile's
// own single-value GFX_TARGET arg) of gfx archs to build separately - one
// full single-arch image per entry, not one fat multi-arch image. Each
// entry must be a target AMD actually publishes apt meta-packages for
// (amdrocm<ver>-<gfx>, amdrocm-core-dev<ver>-<gfx>); see AMD's install
// docs for the current list:
// https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=all&w=compute&os=ubuntu&ubuntu-ver=26.04&i=pkgman#rocm-install-meta-packages
// Deliberately not "build for every arch AMD lists" by default - keeping
// this an explicit, maintained set avoids silently building (and pushing
// build time into) architectures nobody in this homelab actually runs.

variable "ROCM_VERSION" {
  default = "10.0.0"
}

variable "LLAMA_CPP_REF" {
  default = "v0.4.0"
}

variable "GFX_TARGETS" {
  default = "gfx1201"
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

target "_common" {
  context    = "."
  dockerfile = "Dockerfile.rocm-10.0.0.ubuntu26"
  args = {
    LLAMA_CPP_REF = LLAMA_CPP_REF
  }
}

target "bench" {
  name     = "bench-${gfx}"
  matrix   = { gfx = split(",", GFX_TARGETS) }
  inherits = ["_common"]
  target   = "bench"
  args     = { GFX_TARGET = gfx }
  tags     = image_tag("bench", gfx)
}

target "server" {
  name     = "server-${gfx}"
  matrix   = { gfx = split(",", GFX_TARGETS) }
  inherits = ["_common"]
  target   = "server"
  args     = { GFX_TARGET = gfx }
  tags     = image_tag("server", gfx)
}

target "light" {
  name     = "light-${gfx}"
  matrix   = { gfx = split(",", GFX_TARGETS) }
  inherits = ["_common"]
  target   = "light"
  args     = { GFX_TARGET = gfx }
  tags     = image_tag("light", gfx)
}

group "default" {
  targets = ["bench", "server", "light"]
}
