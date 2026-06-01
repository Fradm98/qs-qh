#!/usr/bin/env bash
set -ex

jupyter nbconvert \
  --ExecutePreprocessor.allow_errors=True \
  --ExecutePreprocessor.timeout=-1 \
  --FilesWriter.build_directory=../results \
  --execute 1string_figs.ipynb \
  --to html
 
jupyter nbconvert \
  --ExecutePreprocessor.allow_errors=True \
  --ExecutePreprocessor.timeout=-1 \
  --FilesWriter.build_directory=../results \
  --execute 3string_figs.ipynb \
  --to html

jupyter nbconvert \
  --ExecutePreprocessor.allow_errors=True \
  --ExecutePreprocessor.timeout=-1 \
  --FilesWriter.build_directory=../results \
  --execute benchmark_fig.ipynb \
  --to html
