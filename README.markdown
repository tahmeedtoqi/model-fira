# FIRA: Transformer-Based Language Model with Mixture-of-Experts 

![FIRA Banner](fira.png)  
*FIRA banner with a black background, a stylized brain icon, and metallic 'FIRA' text, symbolizing AI innovation.*

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org) [![License](https://img.shields.io/badge/license-Magentabits-green)](LICENSE) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org) [![Status](https://img.shields.io/badge/status-In%20Production-yellow)]()

**FIRA** is a transformer-based language model designed for efficiency and scalability, leveraging a **Mixture-of-Experts (MoE)** layer to dynamically route inputs to specialized sub-networks. This approach enables FIRA to achieve greater model capacity with reduced computational overhead, making it a powerful tool for advanced language modeling across multiple sectors.

Developed by **Magentabits**, FIRA is in the **production phase**, with ongoing work to implement **advanced reasoning abilities** for a distilled version, targeting complex tasks like logical reasoning and multi-step problem-solving. FIRA outperforms models like GPT-2 (124M), Google Gemma (2B), and LLaMA (7B) in efficiency and adaptability, as detailed in our [technical paper](FIRA.pdf).

## Project Status

FIRA is actively under development, with key focuses:
- **Production Phase**: Optimizing the MoE architecture for large-scale deployment.
- **Advanced Reasoning**: Building a distilled version with enhanced reasoning for tasks requiring deeper understanding.
- **Evaluation**: Planning benchmark evaluations on standard NLP datasets.
- **Community Engagement**: Seeking contributions to refine and expand FIRA’s capabilities.

Join us in shaping the future of efficient language modeling!

## Features

- **Dynamic Expert Routing**: MoE layer optimizes computation by selecting experts per input.
- **Scalable Architecture**: Supports billions of parameters with minimal compute cost.
- **Efficient Inference**: Activates only necessary experts for real-time performance.
- **Load Balancing**: Ensures stable training with even expert utilization.
- **Custom Tokenizer**: Optimized vocab size of 50,304 for efficiency.
- **Advanced Reasoning (In Development)**: Enhanced capabilities for logical and multi-step tasks.

## Use Cases and Sectors

FIRA’s MoE architecture enables versatile applications across various sectors, as outlined in the [technical paper](FIRA.pdf):

| Sector                    | Use Case                                          | Example Application                              |
|---------------------------|--------------------------------------------------|--------------------------------------------------|
| **NLP Research**          | Large-scale text generation                      | Developing new language models and benchmarks    |
| **Real-Time Applications**| Efficient inference for interactive systems      | Chatbots, virtual assistants, customer support   |
| **Domain-Specific AI**    | Fine-tuning for specialized tasks                | Medical report generation, legal contract analysis |
| **Content Creation & Education** | Automated content and tutoring systems     | Educational tools, automated blog writing        |

These use cases leverage FIRA’s ability to generate coherent text, adapt to niche domains, and perform efficiently in resource-constrained environments.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Dependencies: `safetensors`, `mpi4py`, `torchgpipe`, `tokenizers`, `numpy`, `pandas`, `matplotlib`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/magentabits/fira.git
   cd fira
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Example `requirements.txt`:
   ```
   torch>=2.0.0
   safetensors
   mpi4py
   torchgpipe
   tokenizers
   numpy
   pandas
   matplotlib
   ```

3. Download the dataset (e.g., FineWeb-Edu-10B) and tokenizer:
   ```bash
   mkdir -p /kaggle/input
   # Place dataset and tokenizer.json in /kaggle/input
   ```

## Usage

### Training FIRA
Train FIRA on the FineWeb-Edu-10B dataset:
```bash
python training_with_MoE.py
```

Key hyperparameters:
| Parameter           | Value   | Description                          |
|---------------------|---------|--------------------------------------|
| Batch Size          | 16      | Number of samples per batch          |
| Sequence Length     | 512     | Context length for training          |
| Learning Rate       | 6e-4    | Maximum learning rate with decay     |
| Number of Layers    | 12      | Transformer layers                   |
| Number of Experts   | 4       | Experts in MoE layer                 |
| Dropout             | 0.1     | Dropout rate for regularization      |

### Generating Text
Generate text using a trained model:
```python
from Fira import FIRA
from training_with_MoE import encode, decode

model = FIRA(vocab_size=50304, d_model=768, n_head=12, num_layers=12, d_ff=3072, num_experts=4)
model.load_state_dict(torch.load("path/to/model.safetensors"))
prompt = "Hello, my name is Fira, I am an AI assistant"
generated = decode(model.generate(encode(prompt), max_new_tokens=32, temperature=1.0, top_k=50)[0].tolist())
print(generated)
```

<details>
<summary>Advanced Usage: Pipeline Parallelism</summary>
FIRA supports pipeline parallelism via `torchgpipe`. To enable:
1. Ensure multi-GPU setup (`cuda:0`, `cuda:1`).
2. The `create_pipeline_model` function splits the model across GPUs for efficient training.
3. Check `training_with_MoE.py` for implementation details.
</details>

##  Architecture Overview

![FIRA Architecture](workflow.svg)  

The FIRA model implements a sophisticated transformer architecture with Mixture-of-Experts (MoE) layers for efficient scaling. Below is the complete architectural flow:



*FIRA’s transformer blocks with Multi-Head Self-Attention and Mixture-of-Experts layers.*

FIRA integrates a Mixture-of-Experts (MoE) layer within a transformer decoder framework:
- **Token & Position Embeddings**: Maps inputs to a shared embedding space.
- **Transformer Blocks**: Combines Multi-Head Self-Attention (MHSA) and MoE layers.
- **MoE Layer**: Dynamically routes inputs to expert networks for efficiency.
- **Language Modeling Head**: Outputs logits for next-token prediction.

See the [technical paper](FIRA.pdf) for a detailed mathematical formulation.

## 📊 Model Comparison

FIRA’s MoE architecture offers advantages over traditional transformer models:

| Model             | Parameters | Architecture              | Scaling Approach        | Use Case                     |
|-------------------|------------|---------------------------|-------------------------|------------------------------|
| **FIRA**          | 334.74M    | Transformer with MoE       | Dynamic expert routing  | Efficient language modeling  |
| **GPT-2 (124M)**  | 124M       | Transformer decoder        | Static FFN              | Text generation              |
| **Google Gemma (2B)** | 2B      | Transformer-based          | Traditional scaling     | Multimodal tasks             |
| **LLaMA (7B)**    | 7B         | Optimized transformer      | Efficient training      | Research and development     |

FIRA’s dynamic routing and load balancing enhance efficiency and scalability.

## Training Details

- **Dataset**: FineWeb-Edu-10B
- **Parameters**: ~334.74M
- **Optimizer**: AdamW with cosine annealing
- **Hardware**: Multi-GPU setup with `torchgpipe` for pipeline parallelism
- **Logging**: Progress logged to `training_log.csv` and visualized in `training_progress.png`

## Roadmap

FIRA’s development roadmap includes:
- **Q3 2025**: Complete advanced reasoning for distilled version, targeting logical and multi-step tasks.
- **Q4 2025**: Conduct benchmark evaluations on standard NLP datasets.
- **Q1 2026**: Explore open-source release of select components.
- **Ongoing**: Optimize inference for edge devices and expand domain-specific fine-tuning.

##  Acknowledgments

We thank:
- **M.M. Tahmeed Thoky** and **M.D. Ibne Jihan** for leading the project.
- The **PyTorch** and **torchgpipe** communities for enabling efficient model development.
- The **FineWeb-Edu-10B** dataset creators for providing high-quality training data.
- Our team at **Magentabits** for their dedication to advancing AI.

## 🤝 Contributing

We welcome contributions to enhance FIRA! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Interested in advanced reasoning or domain-specific applications? Join us! Follow our [code of conduct](CODE_OF_CONDUCT.md).

## 📜 License

FIRA is copyrighted by **Magentabits**. See the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions or collaboration, reach out to:
- **M.M. Tahmeed Thoky** (Co-CEO, Magentabits)
- **M.D. Ibne Jihan** (CEO, Magentabits)
- Email: [thoky@magentabits.com](mailto:contact@magentabits.com)

---

*Revolutionize language modeling with FIRA’s MoE architecture! 🌌*