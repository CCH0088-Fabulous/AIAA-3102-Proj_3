#!/usr/bin/env bash
set -euo pipefail

phase_slug="${1:-part3}"
archive_path="${2:-videos.zip}"
videos_dir="results/videos/${phase_slug}"

if [[ ! -d "${videos_dir}" ]]; then
  echo "Expected video directory not found: ${videos_dir}" >&2
  exit 1
fi

find "${videos_dir}" -maxdepth 1 -type f -name '*.mp4' | grep -q . || {
  echo "No MP4 files found in ${videos_dir}" >&2
  exit 1
}

rm -f "${archive_path}"
zip -j "${archive_path}" "${videos_dir}"/*.mp4

echo "Created ${archive_path} from ${videos_dir}"