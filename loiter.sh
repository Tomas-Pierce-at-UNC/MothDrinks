#!/bin/bash

if [[ $# > 0 ]]; then
	source bin/activate
	python3 -m idlelib.idle "$1"
fi