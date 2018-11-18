# adds the project root to your path
# run with: . /path/to/fake-install.sh
export PROJECT_ROOT="`pwd`/`dirname "$BASH_SOURCE"`/.."
export PATH=$PROJECT_ROOT:$PATH
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
