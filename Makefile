run:
	@python main.py

format:
	black .

clean:
	rm -rf A/__pycache__
	rm -rf B/__pycache__
