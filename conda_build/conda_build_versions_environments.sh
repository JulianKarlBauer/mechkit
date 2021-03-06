#!/bin/sh

# ====================================================
# Please note the bug, that for conda-build the option '--output' does
# not respect the directories given by '--output-folder':
# https://github.com/conda/conda-build/issues/1957
# ====================================================

tmp=$(dirname $(conda-build --output .))
system=$(basename $tmp)
root_dir=$(dirname $tmp)


for py_version in '3.7' '3.8' '3.5' '3.6' '2.7'
do
  package_name=$(basename $(conda-build --python ${py_version} --output .))
  package_path="${root_dir}/${py_version}/${system}/${package_name}"

  conda-build --no-anaconda-upload \
              --python ${py_version} \
              --output-folder "${root_dir}/${py_version}" .
  for platform in 'linux-64' 'linux-32' 'win-32' 'win-64' 'osx-64'
  do
    conda-convert -p ${platform} -o "${root_dir}/${py_version}" ${package_path}
    anaconda upload "${root_dir}/${py_version}/${platform}/${package_name}"
  done
done
