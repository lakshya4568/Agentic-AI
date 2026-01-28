# Lab 1: 5 Levels of Text Splitting

This lab explores one of the most effective strategies to improve the performance of language model applications: **Chunking** (or Text Splitting). It covers five progressive levels of complexity for dividing large data into smaller, manageable pieces.

## Overview

The notebook [Fine Tuning.ipynb](Fine%20Tuning.ipynb) provides a comprehensive guide to understanding chunking theory and practical implementation using LangChain.

## Content Summary

### Level 1: Character Splitting

The most basic form of splitting, dividing text into N-character sized chunks regardless of content. It introduces concepts like **Chunk Size** and **Chunk Overlap**.

### Level 2: Recursive Character Text Splitting

A more sophisticated method that uses a list of separators (like newlines or spaces) to split text recursively, attempting to keep related content together.

### Level 3: Document Specific Splitting

Tailored chunking methods for specific document types, including:

- **PDFs**
- **Markdown**
- **Python Code**

### Level 4: Semantic Splitting

An advanced method that uses embeddings to identify semantic transitions in the text, ensuring that each chunk represents a coherent idea.

### Level 5: Agentic Splitting

An experimental approach that utilizes an AI agent to intelligently determine where to split text based on logical topic boundaries.

## Requirements

- `langchain`
- `langchain-openai`
- `langchain-community`
- `unstructured`
- `pdf2image`
- `pytesseract`
- `tiktoken`

## Usage

Open the [Fine Tuning.ipynb](Fine%20Tuning.ipynb) notebook and follow the step-by-step instructions and code examples for each level of text splitting.
