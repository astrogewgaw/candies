pkg  := "candies"
desc := "Sweet, sweet candy-dates!"

alias c := clean
alias i := install
alias u := uninstall

# Clean up.
@clean:
    echo "Cleaning..."
    rm -rf tmp
    rm -rf dist
    rm -rf build
    rm -rf .eggs
    rm -rf .coverage
    rm -rf .mypy_cache
    rm -rf docs/build/*
    rm -rf .pytest_cache
    fd -I -e pyc -x rm -rf
    fd -I __pycache__ -x rm -rf

# Install.
@install: && clean
    echo "Installing..."
    pip install -e .

# Uninstall.
@uninstall: && clean
    echo "Uninstalling {{pkg}}..."
    pip uninstall {{pkg}}
    rm -rf src/{{pkg}}.egg-info
    rm -rf src/{{pkg}}/_version.py
