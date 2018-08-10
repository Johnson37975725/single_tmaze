#!/usr/bin/sh

TARGET=$(ls | grep py$ | grep -v experiment)
for F in $TARGET; do
  python3 $F && echo "[OK] "$F
done
