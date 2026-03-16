#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# USER CONFIG
# ============================================================

APT_PACKAGES=(
  "aria2"
)

PIP_PACKAGES=(
)

NODES=(
  "https://github.com/ltdrdata/ComfyUI-Manager"
  "https://github.com/cubiq/ComfyUI_essentials"
  "https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet"
  "https://github.com/kijai/ComfyUI-KJNodes"
)

CHECKPOINT_MODELS=(
)

CLIP_VISION_MODELS=(
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors?download=true"
)

UNET_MODELS=(      
)

LORA_MODELS=( 
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/8697fbd00bf062350864a3ff431b077fbc62886d/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors?download=true"
)

VAE_MODELS=(    
  "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors?download=true"
)

UPSCALE_MODELS=(  
  "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth?download=true"
)

CONTROLNET_MODELS=( )

DIFFUSION_MODELS=(
  "https://civitai.com/api/download/models/2745638?type=Model&format=SafeTensor&size=full&fp=bf16"
  "https://civitai.com/api/download/models/2580860?type=Model&format=SafeTensor&size=pruned&fp=bf16"
)

TEXT_ENCODER_MODELS=(
  "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors?download=true"
)

# ============================================================
# DO NOT EDIT BELOW
# ============================================================

log(){ echo "[provision] $*"; }

WORKSPACE="${WORKSPACE:-/workspace}"
COMFY_WORKSPACE="/workspace/ComfyUI"
INTERNAL_COMFY="/opt/workspace-internal/ComfyUI"

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

APT_INSTALL="${APT_INSTALL:-apt-get install -y --no-install-recommends}"

# ---- hardening state ----
NODE_REQ_FAILS=()
MODEL_DL_FAILS=()

# If you want to fail the whole script when any model download fails, set:
#   export FAIL_ON_MODEL_DL=1
FAIL_ON_MODEL_DL="${FAIL_ON_MODEL_DL:-0}"

# Unified HF token env support
get_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "$HF_TOKEN"
    return 0
  fi
  if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    echo "$HUGGINGFACE_HUB_TOKEN"
    return 0
  fi
  echo ""
}

normalize_comfy_paths() {
  if [[ -d "$INTERNAL_COMFY" && -f "$INTERNAL_COMFY/main.py" ]]; then
    ln -sfn "$INTERNAL_COMFY" "$COMFY_WORKSPACE"
    log "Linked $COMFY_WORKSPACE -> $INTERNAL_COMFY"
  fi

  if [[ ! -f "$COMFY_WORKSPACE/main.py" ]]; then
    log "ERROR: ComfyUI not found at $COMFY_WORKSPACE"
    exit 1
  fi
}

pip_install() {
  if [[ -x "$PIP_BIN" ]]; then
    "$PIP_BIN" install --no-cache-dir "$@"
    return 0
  fi

  if [[ -x "$PYTHON_BIN" ]]; then
    "$PYTHON_BIN" -m pip install --no-cache-dir "$@"
    return 0
  fi

  pip install --no-cache-dir "$@"
}

provisioning_get_apt_packages() {
  if [[ ${#APT_PACKAGES[@]} -gt 0 ]]; then
    log "Installing apt packages: ${APT_PACKAGES[*]}"
    if command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo $APT_INSTALL "${APT_PACKAGES[@]}"
    else
      apt-get update
      $APT_INSTALL "${APT_PACKAGES[@]}"
    fi
  fi
}

provisioning_get_pip_packages() {
  if [[ ${#PIP_PACKAGES[@]} -gt 0 ]]; then
    log "Installing pip packages: ${PIP_PACKAGES[*]}"
    pip_install "${PIP_PACKAGES[@]}"
  fi
}

# ============================================================
# HF_TRANSFER SUPPORT (hardened)
# ============================================================

provisioning_enable_hf_transfer() {
  log "Enabling hf_transfer (best-effort)..."
  # Don't let this kill the script
  set +e
  pip_install -q hf_transfer huggingface_hub
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    log "hf_transfer/huggingface_hub install failed (continuing with fallback)."
  else
    export HF_HUB_ENABLE_HF_TRANSFER=1
  fi
}

# Returns 0 on success, 1 on not applicable or failed (WITHOUT killing script)
provisioning_hf_transfer_download() {
  local dir="$1"
  local url="$2"

  if [[ ! "$url" =~ ^https://huggingface\.co/ ]]; then
    return 1
  fi
  if [[ "$url" != *"/resolve/"* ]]; then
    return 1
  fi

  local clean="${url%%\?*}"
  local rest="${clean#https://huggingface.co/}"

  local repo_id="${rest%%/resolve/*}"
  local after="${rest#${repo_id}/resolve/}"
  local rev="${after%%/*}"
  local file_path="${after#${rev}/}"

  if [[ -z "$repo_id" || -z "$rev" || -z "$file_path" || "$file_path" == "$after" ]]; then
    return 1
  fi

  mkdir -p "$dir"
  log "HF (hf_transfer) attempt: repo=$repo_id rev=$rev file=$file_path -> $dir"

  # Run python with set +e so failure doesn't abort whole script (critical)
  set +e
  "$PYTHON_BIN" - <<'PY' "$repo_id" "$rev" "$file_path" "$dir"
import os, sys, shutil
repo_id, rev, file_path, out_dir = sys.argv[1:5]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None

try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    print("[provision] huggingface_hub not available:", e)
    sys.exit(2)

try:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        revision=rev,
        token=token,
        cache_dir="/workspace/.hf_cache",
    )
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.basename(file_path))
    shutil.copy2(local_path, dst)
    print(f"[provision] HF (hf_transfer) downloaded OK -> {dst}")
    sys.exit(0)
except Exception as e:
    print("[provision] HF (hf_transfer) failed:", repr(e))
    sys.exit(1)
PY
  local rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    return 0
  fi
  return 1
}

# ------------------------------------------------------------
# Downloader: Civitai uses Content-Disposition; HF uses basename
# ------------------------------------------------------------
provisioning_download_to_dir() {
  local dir="$1"
  local url="$2"
  mkdir -p "$dir"

  local final_url="$url"
  local auth_header=""

  # 1. 환경변수의 CIVITAI_TOKEN을 그대로 사용해 URL 완성
  if [[ -n "${CIVITAI_TOKEN:-}" ]] && [[ "$url" =~ civitai\.com ]]; then
    if [[ "$url" == *"?"* ]]; then
      final_url="${url}&token=${CIVITAI_TOKEN}"
    else
      final_url="${url}?token=${CIVITAI_TOKEN}"
    fi
  fi

  # 2. Hugging Face 토큰 처리 (헤더 사용)
  local hf_token
  hf_token="$(get_hf_token)"
  if [[ -n "$hf_token" ]] && [[ "$url" =~ huggingface\.co ]]; then
    auth_header="Authorization: Bearer ${hf_token}"
  fi

  log "Downloading into $dir"
  log "  from: $final_url"

  # ---- HuggingFace: hf_transfer 시도 ----
  if [[ "$url" =~ huggingface\.co ]]; then
    if provisioning_hf_transfer_download "$dir" "$final_url"; then
      return 0
    fi
  fi

  # ---- Civitai: aria2 버그를 피하기 위해 curl로 강제 다운로드 ----
  if [[ "$url" =~ civitai\.com ]]; then
    set +e
    (cd "$dir" && curl -fL -OJ "$final_url")
    local rc=$?
    set -e
    return $rc
  fi

  # ---- General/HF fallback ----
  local name="${url%%\?*}"
  name="${name##*/}"

  if [[ -z "$name" || "$name" =~ ^[0-9]+$ ]]; then
    set +e
    if command -v aria2c >/dev/null 2>&1; then
      if [[ -n "$auth_header" ]]; then
        aria2c -x 16 -s 16 -k 1M --content-disposition --header="$auth_header" -d "$dir" "$final_url"
      else
        aria2c -x 16 -s 16 -k 1M --content-disposition -d "$dir" "$final_url"
      fi
      rc=$?
    else
      if [[ -n "$auth_header" ]]; then
        (cd "$dir" && curl -fL -H "$auth_header" -OJ "$final_url")
      else
        (cd "$dir" && curl -fL -OJ "$final_url")
      fi
      rc=$?
    fi
    set -e
    return $rc
  fi

  set +e
  if command -v aria2c >/dev/null 2>&1; then
    if [[ -n "$auth_header" ]]; then
      aria2c -x 16 -s 16 -k 1M --header="$auth_header" -o "$name" -d "$dir" "$final_url"
    else
      aria2c -x 16 -s 16 -k 1M -o "$name" -d "$dir" "$final_url"
    fi
    rc=$?
  elif command -v wget >/dev/null 2>&1; then
    if [[ -n "$auth_header" ]]; then
      wget --header="$auth_header" -O "$dir/$name" "$final_url"
    else
      wget -O "$dir/$name" "$final_url"
    fi
    rc=$?
  else
    if [[ -n "$auth_header" ]]; then
      curl -fL -H "$auth_header" -o "$dir/$name" "$final_url"
    else
      curl -fL -o "$dir/$name" "$final_url"
    fi
    rc=$?
  fi
  set -e
  return $rc
}

provisioning_get_models_dir_urlonly() {
  local dir="$1"; shift || true
  local arr=("$@")
  if [[ ${#arr[@]} -eq 0 ]]; then
    return 0
  fi
  for url in "${arr[@]}"; do
    if ! provisioning_download_to_dir "$dir" "$url"; then
      log "MODEL DOWNLOAD FAILED: $url"
      MODEL_DL_FAILS+=("$url")
      if [[ "$FAIL_ON_MODEL_DL" == "1" ]]; then
        log "FAIL_ON_MODEL_DL=1 -> exiting due to model download failure."
        exit 1
      fi
    fi
  done
}

provisioning_get_nodes() {
  local nodes_dir="${COMFY_WORKSPACE}/custom_nodes"
  mkdir -p "$nodes_dir"

  for repo in "${NODES[@]}"; do
    local dir="${repo##*/}"
    local path="${nodes_dir}/${dir}"
    local requirements="${path}/requirements.txt"

    if [[ -d "$path/.git" ]]; then
      log "Updating node: $repo"
      git -C "$path" pull --ff-only || true
    else
      log "Cloning node: $repo"
      git clone --depth=1 --recursive "$repo" "$path"
    fi

    if [[ -f "$requirements" ]]; then
      log "Installing requirements: $requirements"
      set +e
      pip_install -r "$requirements"
      local rc=$?
      set -e
      if [[ $rc -ne 0 ]]; then
        log "Node requirements FAILED: $repo"
        NODE_REQ_FAILS+=("$repo")
      fi
    fi
  done
}

print_summary() {
  if [[ ${#NODE_REQ_FAILS[@]} -gt 0 ]]; then
    log "---- Node requirements failures ----"
    for x in "${NODE_REQ_FAILS[@]}"; do
      log "  - $x"
    done
  fi

  if [[ ${#MODEL_DL_FAILS[@]} -gt 0 ]]; then
    log "---- Model download failures ----"
    for x in "${MODEL_DL_FAILS[@]}"; do
      log "  - $x"
    done
  fi
}

provisioning_start() {
  normalize_comfy_paths

  provisioning_get_apt_packages
  provisioning_get_nodes
  provisioning_get_pip_packages

  provisioning_enable_hf_transfer

  provisioning_get_models_dir_urlonly "${COMFY_WORKSPACE}/models/checkpoints"      "${CHECKPOINT_MODELS[@]}"
  provisioning_get_models_dir_urlonly "${COMFY_WORKSPACE}/models/unet"             "${UNET_MODELS[@]}"
  provisioning_get_models_dir_urlonly "${COMFY_WORKSPACE}/models/loras"            "${LORA_MODELS[@]}"
  provisioning_get_models_dir_urlonly "${COMFY_WORKSPACE}/models/controlnet"       "${CONTROLNET_MODELS[@]}"
  provisioning_get_models_dir_urlonly "${COMFY_WORKSPACE}/models/vae"              "${VAE_MODELS[@]}"
  provisioning_get_models_dir_urlonly "${COMFY_WORKSPACE}/models/upscale_models"   "${UPSCALE_MODELS[@]}"
  provisioning_get_models_dir_urlonly "${COMFY_WORKSPACE}/models/diffusion_models" "${DIFFUSION_MODELS[@]}"
  provisioning_get_models_dir_urlonly "${COMFY_WORKSPACE}/models/text_encoders"    "${TEXT_ENCODER_MODELS[@]}"
  provisioning_get_models_dir_urlonly "${COMFY_WORKSPACE}/models/clip_vision"      "${CLIP_VISION_MODELS[@]}"
  print_summary
  log "Provisioning complete."
}

provisioning_start
