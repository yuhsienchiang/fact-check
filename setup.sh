PROJECT_NAME="fact-check"
PIP_REQUIREMENTS="requirements.txt"

echo "Install project required packages"
if [ -n "$PIP_REQUIREMENTS" ]; then
    pip install -r $PIP_REQUIREMENTS --quiet
fi

echo "Configuring PYTHONPATH for the project"
PYTHON_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p $PYTHON_SITE
cat >> $PYTHON_SITE/$PROJECT_NAME.pth <<EOF
$PWD/src
EOF