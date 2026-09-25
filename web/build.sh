#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build the browser runtime using an already prepared toolchain (see README.md).

bash web/build.sh --model FILE --source-dir DIR --protoc FILE --js-package DIR
                 [--output dist] [--work-dir .web-build] [--build-dir DIR] [--jobs N]

Required: complete ONNX model, ORT 1.29.0 checkout with Emscripten 4.0.23,
native protoc 3.21.12, Node.js 20+, npm 10+ alongside Node.js, and installed
onnxruntime-web@1.29.0 npm package.
Paths are relative to your current directory. Build directory defaults to WORK/build.
No toolchain installation is performed. CMake may fetch pinned C++ dependencies.
EOF
}
fail() { echo "Error: $*" >&2; exit 1; }

model= source_dir= protoc= js_package= build_dir= jobs=
output=dist work_dir=.web-build
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --model|--source-dir|--protoc|--js-package|--output|--work-dir|--build-dir|--jobs)
      (($# >= 2)) && [[ -n $2 && $2 != --* ]] || fail "Missing value for $1"
      key=${1#--}; key=${key//-/_}
      printf -v "$key" '%s' "$2"
      shift 2 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done
[[ -n $model && -n $source_dir && -n $protoc && -n $js_package ]] || {
  usage >&2; fail "--model, --source-dir, --protoc and --js-package are required"
}
for tool in uv git cmake ninja node; do
  command -v "$tool" >/dev/null || fail "Required tool not on PATH: $tool"
done
node -e 'if (+process.versions.node.split(".")[0] < 20) process.exit(1)' || fail "Node.js 20+ required"
npm_bin="$(dirname -- "$(command -v node)")/npm"
[[ -x $npm_bin ]] || fail "npm 10+ must be installed alongside Node.js: $npm_bin"
npm_version=$("$npm_bin" --version) || fail "Could not run npm: $npm_bin"
[[ $npm_version =~ ^([0-9]+)\. ]] && (( BASH_REMATCH[1] >= 10 )) || fail "npm 10+ required; found $npm_version"

web_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(dirname -- "$web_dir")
# Use the existing project environment without installing or syncing anything.
python=$(uv run --locked --no-sync --project "$project_dir" python -c 'import sys; print(sys.executable)')
resolve() { "$python" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1"; }
model=$(resolve "$model")
source_dir=$(resolve "$source_dir")
protoc=$(resolve "$protoc")
js_package=$(resolve "$js_package")
output=$(resolve "$output")
work_dir=$(resolve "$work_dir")
build_dir=$(resolve "${build_dir:-$work_dir/build}")
[[ $source_dir$build_dir != *[[:space:]]* ]] || fail "The ORT checkout and build directory paths must not contain spaces"
jobs=${jobs:-$("$python" -c 'import os; print(min(6, os.cpu_count() or 1))')}
[[ $jobs =~ ^[1-9][0-9]*$ ]] || fail "--jobs must be positive"
[[ $output != / ]] || fail "--output cannot be the filesystem root"
for path in "$model" "$work_dir" "$build_dir" "$source_dir" "$protoc" "$js_package" "$web_dir"; do
  [[ $path != "$output" && $path != "$output/"* ]] || fail "Build inputs and caches must be outside --output"
done
[[ -f $model ]] || fail "Model not found: $model"

ort_version=1.29.0
ort_commit=2e2543fbe9fae542f921d47a72d21d5a4ef0b710
emsdk_version=4.0.23
[[ -f $source_dir/VERSION_NUMBER ]] || fail "Missing prepared ORT checkout: $source_dir"
[[ $(cat "$source_dir/VERSION_NUMBER") == "$ort_version" &&
   $(git -C "$source_dir" rev-parse HEAD) == "$ort_commit" ]] || fail "Expected ORT $ort_version ($ort_commit)"
[[ -x $protoc ]] || fail "Missing native protoc: $protoc"
[[ $("$protoc" --version) == 'libprotoc 3.21.12' ]] || fail "Expected native protoc 3.21.12"
package_version=$(node -p 'JSON.parse(require("fs").readFileSync(process.argv[1])).version' "$js_package/package.json")
[[ $package_version == "$ort_version" ]] || fail "Expected onnxruntime-web@$ort_version"
[[ -f $js_package/dist/ort.wasm.min.mjs ]] || fail "Missing JavaScript WASM API in --js-package"
emscripten="$source_dir/cmake/external/emsdk/upstream/emscripten"
[[ -f $emscripten/emcc.py ]] || fail "Emscripten is not installed; see README.md"
compiler_version=$("$python" "$emscripten/emcc.py" --version)
[[ $compiler_version == *" $emsdk_version "* ]] || fail "Expected Emscripten $emsdk_version"
mkdir -p -- "$work_dir" "$build_dir/MinSizeRel"
echo 'Deriving kernel/type requirements from the complete ONNX model...'
"$python" "$web_dir/required_ops.py" --model "$model" --work-dir "$work_dir" --ort-version "$ort_version"
config="$work_dir/required_operators.config"
build="$build_dir/MinSizeRel"
# ORT deletes its generated headers on every reduction. Preserve them on warm builds.
signature=$("$python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read() + sys.argv[2].encode()).hexdigest())' "$config" "$ort_commit:types:extended:v1")
stamp="$build/swiftf0-reduction.sha256"
if [[ -f $stamp && $(cat "$stamp") == "$signature" &&
      -f $build/op_reduction.generated/onnxruntime/core/providers/op_kernel_type_control_overrides.inc ]]; then
  echo 'Reusing operator reduction.'
else
  "$python" "$source_dir/tools/ci_build/reduce_op_kernels.py" "$config" \
    --cmake_build_dir "$build" --enable_type_reduction --is_extended_minimal_build_or_higher
  printf '%s\n' "$signature" > "$stamp"
fi

prefix_map="-ffile-prefix-map=$source_dir=onnxruntime -ffile-prefix-map=$build_dir=build"
echo 'Building the model-specific ONNX-capable WASM runtime...'
cmake -S "$source_dir/cmake" -B "$build" -G Ninja -C "$web_dir/settings.cmake" \
  --compile-no-warning-as-error -DCMAKE_BUILD_TYPE=MinSizeRel \
  "-DPython_EXECUTABLE=$python" \
  "-DCMAKE_TOOLCHAIN_FILE=$emscripten/cmake/Modules/Platform/Emscripten.cmake" \
  "-DONNX_CUSTOM_PROTOC_EXECUTABLE=$protoc" \
  "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=$web_dir/math.cmake" \
  "-DCMAKE_C_FLAGS=$prefix_map" "-DCMAKE_CXX_FLAGS=$prefix_map"
cmake --build "$build" --target onnxruntime_webassembly --parallel "$jobs"

mkdir -p -- "$(dirname -- "$output")"
stage=$(mktemp -d "$(dirname -- "$output")/.swiftf0-web.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT
cp -- "$work_dir/model.onnx" "$stage/model.onnx"
cp -- "$js_package/dist/ort.wasm.min.mjs" "$web_dir/predict.mjs" "$stage/"
cp -- "$build/ort-wasm-simd-threaded.mjs" "$build/ort-wasm-simd-threaded.wasm" "$stage/"
cp -- "$source_dir/LICENSE" "$source_dir/ThirdPartyNotices.txt" "$stage/"

# Publish after compilation and asset copying succeed. Keep unrelated files.
mkdir -p -- "$output"
for file in "$stage"/*; do
  [[ ! -d $output/$(basename -- "$file") ]] || fail "Output asset is a directory: $file"
done
for file in "$stage"/*; do mv -f -- "$file" "$output/"; done
echo "Saved $output. Serve these files over HTTP(S) with server/CDN Brotli or gzip compression."
