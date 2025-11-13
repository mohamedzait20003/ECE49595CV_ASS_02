# Makefile for ECE 49595CV Assignment 2: ResNet-improved AlexNet

.PHONY: all install run notebook clean test lint

# Default target
all: run

# Install Python dependencies
install:
	pip install -r requirements.txt

# Run the ResNet-improved CNN classifier  
run:
	python main.py

# Launch Jupyter notebook for interactive training
notebook:
	jupyter notebook Notebooks/notebook.ipynb

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf Model/__pycache__ Utilities/__pycache__
	rm -f *.pth model_*.pth checkpoint_*.pth

# Test dependencies
test:
	python -c "import torch, torchvision, PIL, numpy, tqdm, matplotlib; print('✓ All dependencies available')"
	python -c "print('PyTorch version:', torch.__version__)"
	python -c "print('CUDA available:', torch.cuda.is_available())"

# Lint Python code
lint:
	flake8 --max-line-length=88 --ignore=E203,W503 *.py Model/ Utilities/

# Help
help:
	@echo "Makefile for ECE 49595CV Assignment 2"
	@echo "ResNet-improved AlexNet CNN Classifier"
	@echo ""
	@echo "Available targets:"
	@echo "  make install   - Install Python dependencies"
	@echo "  make run       - Run the CNN classifier training"
	@echo "  make notebook  - Launch interactive Jupyter notebook"
	@echo "  make clean     - Remove generated files and caches"
	@echo "  make test      - Test dependencies and CUDA availability"
	@echo "  make lint      - Run code style checks"
	@echo "  make help      - Show this help message"
