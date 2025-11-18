#!/bin/bash

# Script to remove large file from Git history

echo "Removing data/cifar-10-python.tar.gz from Git history..."

# Remove from all branches and tags
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch 'data/cifar-10-python.tar.gz'" \
  --prune-empty --tag-name-filter cat -- --all

echo ""
echo "Cleaning up backup refs..."
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

echo ""
echo "Running garbage collection..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "Done! The file has been removed from Git history."
echo "You can now force push with: git push --force --all"

