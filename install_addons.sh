#!/bin/bash
set -e

UNAME=`uname -m`

if [ "$UNAME" == "aarch64" ]; then
	echo ""
	echo $PWD
	echo ""

	git clone --branch v0.18.0 https://github.com/tensorflow/addons.git

	# ls $app > filename.txt
	# cat filename.txt

	cd /app/addons
	python3 ./configure.py
	bazel build build_pip_pkg
	bazel-bin/build_pip_pkg /tmp/tensoraddons_pkg

	pip3 install /tmp/tensoraddons_pkg/tensorflow_addons-0.18.0-cp38-cp38-linux_aarch64.whl

else
	echo $PWD
fi
