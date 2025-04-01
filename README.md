# Traditional Music Generator

Welcome to the Traditional Music Generator project! This repository leverages Python-based generative AI methods (inspired by image-generation techniques) and applies them to create distinctive musical pieces reminiscent of traditional styles or instruments.

## Overview
This project explores how generative models can be adapted from the image-generation domain to produce engaging and culturally rich music. By training on curated musical datasets, it aims to capture elements such as melody, rhythm, and timbre associated with specific music traditions.

## Features
- Utilizes Python-based scripts to preprocess data, train models, and generate music.
- Supports various generative architectures (e.g., sequence models, diffusion-based).
- Offers configuration options for tuning training parameters and output length.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/FPT-KhoiLe/Traditional_Music_Generator.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your dataset by placing audio files in the specified directory or using the provided scripts for data collection.
4. Start training the model:
   ```bash
   python train.py --config config.yaml
   ```
5. Generate your own music samples:
   ```bash
   python generate.py --model_path model_checkpoint.pt
   ```

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request for improvements such as:
- Additional training approaches
- Enhanced data preprocessing
- Extended musical style coverage
- Performance optimizations

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it in accordance with the license terms.

## Contact
For any inquiries or suggestions, please open an issue in this repository or reach out to the owner.
