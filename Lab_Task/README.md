# Lab Task: Fine-tuning Small Language Model

## Overview

This folder contains the implementation of fine-tuning a Small Language Model (SLM) as part of the Agentic AI course lab task.

## Task Description

**Objective:** Fine-tune a Small Language Model on text data using Google Colab

**Requirements:**
- Use Google Colab for training
- Choose a text dataset from Hugging Face (unique dataset)
- Select an SLM with less than 3B parameters
- Fine-tune and evaluate the model
- Document steps, results, and observations

## Implementation Details

### Model: Qwen2-0.5B-Instruct
- **Parameters:** ~500M (well under 3B limit)
- **Architecture:** Transformer-based causal language model
- **Training Method:** QLoRA (Quantized Low-Rank Adaptation)
- **Quantization:** 4-bit (NF4) for memory efficiency

### Dataset: medical_meadow_medqa
- **Source:** `medalpaca/medical_meadow_medqa` from Hugging Face
- **Domain:** Medical Question Answering
- **Size:** ~10,000 QA pairs
- **Format:** Instruction-response pairs

### Training Configuration

```python
# Hardware
- GPU: Google Colab T4 (16GB VRAM)
- Precision: FP16 + 4-bit quantization

# Hyperparameters
- Epochs: 3
- Batch Size: 4 (with gradient accumulation x4)
- Learning Rate: 2e-4
- Optimizer: Paged AdamW 8-bit
- Max Sequence Length: 512

# LoRA Config
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target Modules: q_proj, k_proj, v_proj, o_proj
```

## Files

- **Fine_tuning_Lab_task1.ipynb**: Complete notebook with all code, explanations, and results
- **README.md**: This file

## Running the Notebook

### Step 1: Open in Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" → "Open notebook" → "GitHub"
3. Enter: `https://github.com/lakshya4568/Agentic-AI`
4. Select: `Lab_Task/Fine_tuning_Lab_task1.ipynb`

### Step 2: Enable GPU

1. Click "Runtime" → "Change runtime type"
2. Select "T4 GPU" from Hardware accelerator dropdown
3. Click "Save"

### Step 3: Run the Notebook

- Run cells sequentially from top to bottom
- Total runtime: 30-60 minutes (depending on dataset size)
- The notebook is self-contained with all necessary code

## Key Features

### Memory Efficiency
- **4-bit Quantization:** Reduces model size by ~75%
- **QLoRA:** Trains only 0.5-1% of parameters
- **Gradient Checkpointing:** Saves memory during backpropagation
- **Paged Optimizers:** Handles memory spikes efficiently

### Training Features
- **Supervised Fine-Tuning (SFT):** Task-specific training
- **Cosine Learning Rate Schedule:** Smooth convergence
- **Early Stopping:** Prevents overfitting
- **Automatic Mixed Precision:** Faster training

### Evaluation
- **Training Loss:** Monitors learning progress
- **Evaluation Loss:** Validates on held-out data
- **Perplexity:** Measures model uncertainty
- **Inference Testing:** Real-world response quality

## Expected Results

### Training Metrics
- Initial Loss: ~2.5-3.0
- Final Loss: ~0.5-1.5
- Perplexity: 5-30 (lower is better)

### Model Performance
- Coherent medical responses
- Improved domain knowledge
- Better instruction following
- Structured and accurate answers

## Technical Stack

```python
Libraries:
├── transformers      # Model loading and training
├── datasets          # Data loading and processing
├── peft              # Parameter-efficient fine-tuning (LoRA)
├── bitsandbytes      # Quantization
├── trl               # Supervised fine-tuning trainer
├── torch             # Deep learning framework
└── accelerate        # Distributed training utilities
```

## Observations

### Advantages of QLoRA
1. **Memory Efficient:** Fits large models on limited hardware
2. **Fast Training:** Only updates small adapter layers
3. **Preserves Knowledge:** Base model weights remain frozen
4. **Easy Deployment:** Can merge or swap adapters

### Dataset Selection
- **Medical domain** chosen for practical relevance
- **QA format** aligns with instruction-following tasks
- **Sufficient size** for meaningful fine-tuning
- **High quality** data ensures better results

### Model Choice
- **Qwen2-0.5B** balances performance and efficiency
- **Instruction-tuned** base makes adaptation easier
- **Open-source** allows full control and customization
- **Active community** provides support and resources

## Potential Improvements

1. **Training:**
   - Increase epochs (5-10) for better convergence
   - Experiment with LoRA rank (8, 32, 64)
   - Try different learning rates (1e-4, 5e-4)
   - Add warmup steps for stability

2. **Evaluation:**
   - Add BLEU score for generation quality
   - Calculate ROUGE for summarization tasks
   - Implement custom medical accuracy metrics
   - Test on out-of-domain questions

3. **Data:**
   - Augment with more medical datasets
   - Balance dataset across specialties
   - Filter low-quality examples
   - Add adversarial examples for robustness

4. **Model:**
   - Try larger models (1.5B, 2.7B)
   - Experiment with different architectures
   - Compare multiple base models
   - Test different quantization methods

## Resources

### Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Qwen2 Model Card](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)

### Tutorials
- [Fine-tuning LLMs with QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft/tutorial/peft_model_config)
- [SFT Trainer Guide](https://huggingface.co/docs/trl/sft_trainer)

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM):**
   - Reduce batch size to 2
   - Decrease max_seq_length to 256
   - Enable gradient checkpointing
   - Use smaller LoRA rank

2. **Slow Training:**
   - Ensure GPU is enabled
   - Check FP16 is activated
   - Reduce logging frequency
   - Use smaller evaluation set

3. **Poor Results:**
   - Increase training epochs
   - Lower learning rate
   - Check data formatting
   - Validate dataset quality

4. **Import Errors:**
   - Reinstall packages with `!pip install -U`
   - Restart runtime
   - Check package compatibility
   - Use specific versions if needed

## Citation

If using this work, please cite:

```bibtex
@misc{sharma2026finetuning,
  author = {Sharma, Lakshya},
  title = {Fine-tuning Small Language Models for Medical QA},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/lakshya4568/Agentic-AI}}
}
```

## Contact

**Author:** Lakshya Sharma  
**Institution:** Sharda University  
**Course:** Agentic AI  
**Section:** F (Group 2)

---

**Last Updated:** February 11, 2026