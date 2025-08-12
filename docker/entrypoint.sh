#!/bin/bash

# Docker entrypoint script for TMRT
set -e

echo "=== TMRT Container Starting ==="
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Available commands:"
echo "  demo-toy     - Run toy model demo"
echo "  demo-unicode - Test Unicode mutations only"
echo "  demo-scaffold - Test role scaffolding only"
echo "  search       - Run full evolutionary search"
echo "  notebook     - Start Jupyter notebook server"

# Parse command
case "${1:-demo-toy}" in
    "demo-toy")
        echo "Running toy model demo..."
        python -m tmrt.demo --mode full --config configs/toy_demo.yaml
        ;;
    "demo-unicode")
        echo "Testing Unicode mutations..."
        python -m tmrt.demo --mode unicode
        ;;
    "demo-scaffold")
        echo "Testing role scaffolding..."
        python -m tmrt.demo --mode scaffold
        ;;
    "search")
        CONFIG=${2:-configs/full_search.yaml}
        SEED=${3:-42}
        echo "Running evolutionary search with config: $CONFIG"
        python -c "
from tmrt import SearchController, load_config
config = load_config('$CONFIG')
controller = SearchController(config['model_name'], config, seed=$SEED)
results = controller.run_search()
print(f'Search completed. Best fitness: {results[\"best_fitness\"]:.4f}')
        "
        ;;
    "notebook")
        echo "Starting Jupyter notebook server..."
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app
        ;;
    *)
        echo "Running custom command: $@"
        exec "$@"
        ;;
esac

echo "=== TMRT Container Complete ==="
