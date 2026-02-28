#!/bin/bash

# Build documentation
sphinx-build -E -b html -d build/doctrees . build/html
