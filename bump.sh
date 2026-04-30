#!/bin/bash
# bump.sh — bump VERSION before kicking off a new training run.
#
# Workflow rule: every retrain bumps the patch version. Reasons:
#   - pointer_model_v{VERSION}_{err}px.pt files don't collide between runs
#   - git history maps cleanly to model lineage (each tag = a trained model)
#   - we can compare any two trained variants without losing weights
#
# Usage:
#   ./bump.sh                  # patch bump (default — for retrains)
#   ./bump.sh patch            # patch bump (e.g. 0.3.1 → 0.3.2)
#   ./bump.sh minor            # minor bump (e.g. 0.3.1 → 0.4.0; reset patch)
#   ./bump.sh major            # major bump (e.g. 0.3.1 → 1.0.0; reset minor+patch)
#   ./bump.sh patch --commit   # bump + git add VERSION + commit + tag v{NEW}
#
# Bump levels (project convention, slightly tighter than vanilla semver):
#   patch — retrain with same/similar code, hyperparam tweak
#   minor — substantive algorithm or data change (new loss, new aug, new arch)
#   major — breaking interface change (e.g. coord-space change for click_at)

set -e
cd "$(dirname "$0")"

LEVEL="${1:-patch}"
COMMIT="${2:-}"

CURRENT=$(tr -d '\n\r' < VERSION)
IFS=. read -r MAJ MIN PAT <<< "$CURRENT"

case "$LEVEL" in
    patch) PAT=$((PAT + 1)) ;;
    minor) MIN=$((MIN + 1)); PAT=0 ;;
    major) MAJ=$((MAJ + 1)); MIN=0; PAT=0 ;;
    *) echo "usage: bump.sh [patch|minor|major] [--commit]" >&2; exit 1 ;;
esac

NEW="$MAJ.$MIN.$PAT"
echo "$NEW" > VERSION
echo "VERSION: $CURRENT → $NEW"

if [ "$COMMIT" = "--commit" ]; then
    git add VERSION
    git commit -m "bump to v$NEW"
    git tag "v$NEW" -m "v$NEW"
    echo "git: committed + tagged v$NEW"
else
    echo "tip: run \`git add VERSION && git commit -m 'bump to v$NEW' && git tag v$NEW\`"
    echo "or:  \`./bump.sh $LEVEL --commit\` next time"
fi
