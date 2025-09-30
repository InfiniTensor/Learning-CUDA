
OUTPUT_PATH := output/


.PHONY: all build-kernel run-kernel generate-file


all: build-kernel run-kernel

build-kernel: $(OUTPUT_PATH)unique_extraction

run-kernel: $(OUTPUT_PATH)unique_extraction
	./$(OUTPUT_PATH)unique_extraction

generate-file: $(OUTPUT_PATH)generate_file
	./$(OUTPUT_PATH)generate_file $(size)

$(OUTPUT_PATH): 
	mkdir -p $@

$(OUTPUT_PATH)unique_extraction: src/unique_extraction.cu
	nvcc -o $@ $^

$(OUTPUT_PATH)generate_file: tester/generate_file.cpp
	g++ -o $@ $^