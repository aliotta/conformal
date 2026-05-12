.PHONY: install run-projector run-gif-gen

# Target to install dependencies
install:
	pip3 install -r requirements.txt

# Target to run the interactive projector
run-projector:
	python3 ./scripts/mlx_transform.py

# Target to generate the loop GIFs
run-gif-gen:
	python3 ./scripts/gif_gen.py