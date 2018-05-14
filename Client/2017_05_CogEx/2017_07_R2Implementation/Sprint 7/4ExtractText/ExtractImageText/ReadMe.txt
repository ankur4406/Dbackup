Purpose: To create an algorithm for parsing text from an unstructured scanned document (image)


Inputs: 1. Scanned image document
          
	2. Pre-trained classification model for document body identification


Outputs: A json file in a pre-defined format containing the converted text along with other requisite information


Test Cases:
	python3 ExtractImageText.py Sample1.png
	python3 ExtractImageText.py Sample2.png