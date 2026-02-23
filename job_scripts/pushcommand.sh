git submodule foreach --recursive '
  if git status --porcelain | grep -q .; then
    git add -A
    git commit -m "Update submodule" || true
    git push
  fi
'