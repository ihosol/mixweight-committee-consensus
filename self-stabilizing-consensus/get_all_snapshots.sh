#!/usr/bin/env bash
set -euo pipefail

MONTH=2025-09           # YYYY-MM you want
WORKFLOW="daily.yml"    # name of the workflow that produced the data
BRANCH="main"           # branch the workflow ran on
OUT_ROOT="SNAPSHOTS"         # where to unpack snapshots locally

run_ids=$(gh run list \
    --workflow "$WORKFLOW" \
    --branch "$BRANCH" \
    --json databaseId,createdAt \
    --limit 200 |
    jq -r --arg month "$MONTH" '
      .[] | select(.createdAt | startswith($month)) | .databaseId
    ')

if [[ -z "$run_ids" ]]; then
    echo "No runs found for $MONTH on $BRANCH/$WORKFLOW" >&2
    exit 1
fi

for run_id in $run_ids; do
    echo "==> Processing run $run_id"
    run_info=$(gh run view "$run_id" --json createdAt --jq '.createdAt')
    day=${run_info%%T}                # 2025-09-01T... -> 2025-09-01
    target_dir="$OUT_ROOT/$day"
    mkdir -p "$target_dir/raw"

    # list artifacts for the run (often there’s a single ZIP per day)
    gh api \
      repos/:owner/:repo/actions/runs/"$run_id"/artifacts \
      --jq '.artifacts[] | {id: .id, name: .name}' |
    jq -c '.' |
    while IFS= read -r artifact; do
        art_id=$(jq -r '.id' <<<"$artifact")
        art_name=$(jq -r '.name' <<<"$artifact")
        zip_path="$target_dir/raw/${art_name}.zip"

        echo "   downloading artifact $art_name -> $zip_path"
        gh api \
          -H "Accept: application/vnd.github+json" \
          repos/:owner/:repo/actions/artifacts/"$art_id"/zip > "$zip_path"

        echo "   unpacking into $target_dir"
        unzip -q -o "$zip_path" -d "$target_dir"
    done
done

echo "All artifacts for $MONTH downloaded under $OUT_ROOT/"
