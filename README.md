# 505-rico-blip-ui-caption
# Multimodal Fine-Tuning of BLIP on RICO-Screen2Words

## Overview

This project demonstrates **multimodal fine-tuning of a Small Language Model (SLM)** using the **RICO-Screen2Words dataset**. The objective is to train a vision-language model capable of generating natural language descriptions for mobile UI screenshots.

The model learns to interpret visual UI components (dialogs, pop-ups, buttons, navigation elements) and generate meaningful textual descriptions of the interface.

This implementation was developed for the **Orange Problem assignment**, which focuses on multimodal fine-tuning using small language models.

---

# Dataset

Dataset used:

**RICO-Screen2Words**

https://huggingface.co/datasets/rootsautomation/RICO-Screen2Words

The dataset contains mobile application screenshots paired with human-written captions describing the interface.

Each data point contains:

- **image** → mobile UI screenshot
- **captions** → multiple textual descriptions of the UI

Example:

Image → screenshot of a calendar popup

Captions:

- "notification displaying a calendar with done option"
- "pop up displaying a calendar"
- "pop up displaying calendar to select date"

For training, the **first caption was selected as the ground truth label**.

This transformation simplifies training while preserving the semantic meaning of the UI.

---

# Model

Model used:

**BLIP Image Captioning Base**

https://huggingface.co/Salesforce/blip-image-captioning-base

BLIP is a **vision-language transformer model** capable of generating captions from images.

Reasons for selecting this model:

- Lightweight enough to run on **T4 GPU (16GB VRAM)**
- Strong baseline for image captioning tasks
- Fully compatible with Hugging Face Transformers
- Suitable for multimodal fine-tuning

BLIP accepts **visual inputs and generates textual outputs**, making it well suited for UI caption generation.

---

# Hugging Face Model

Fine-tuned model repository:

https://huggingface.co/preksham2004/rico-blip-ui-caption

This repository contains:

- fine-tuned model weights
- processor configuration
- tokenizer
- inference-ready model

Users can directly download and run the model for caption generation.

---

# Training Setup

Hardware used:

- **Kaggle GPU (T4 – 16GB VRAM)**

Training configuration:

| Parameter | Value |
|----------|------|
| Batch Size | 4 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| Epochs | 1 |
| Training Samples | 800 |

The dataset was shuffled and a subset of **800 samples** was used to ensure efficient training within GPU memory limits.

This ensures the entire pipeline runs successfully on **T4 compute**, as required by the assignment.

---

# Data Preprocessing

The dataset originally contains **multiple captions per image**.

Example:

captions = [ "notification displaying a calendar with done option", "pop up displaying a calendar", "pop up displaying calendar to select date" ]
To simplify training, the **first caption was used as the label**.

caption = captions[0]
Images were processed using the BLIP processor which performs:

- image resizing
- normalization
- tokenization

This ensures compatibility with the BLIP model architecture.

---

# Fine-Tuning Procedure

The following steps were used to fine-tune the model:

1. Load the dataset from Hugging Face
2. Shuffle the dataset
3. Select a subset of training samples
4. Extract one caption per image
5. Create a PyTorch DataLoader
6. Tokenize images and captions using the BLIP processor
7. Fine-tune the model using cross-entropy loss
8. Save the trained model
9. Upload the model to Hugging Face

The training was implemented using **PyTorch and Hugging Face Transformers**.

---

# Running the Training Notebook

The project was implemented using a **single Kaggle notebook**.

Steps to run:

1. Enable GPU (T4 recommended)
2. Install dependencies
3. Run the notebook cells sequentially

Install dependencies:

pip install transformers datasets pillow accelerate
---

# Loading the Model from Hugging Face

The following code demonstrates how to **download the model from Hugging Face and run inference**.

```python
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

model_name = "preksham2004/rico-blip-ui-caption"

processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image = Image.open("example_ui.png")

inputs = processor(
    images=image,
    return_tensors="pt"
).to(device)

output = model.generate(**inputs)

caption = processor.decode(output[0], skip_special_tokens=True)

print("Generated caption:", caption)
Example output:
pop up displaying calendar to select date
