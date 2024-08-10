# DermaDetectAI

![GitHub Created At](https://img.shields.io/github/created-at/RanitManik/DermaDetectAI)
![GitHub repo size](https://img.shields.io/github/repo-size/RanitManik/DermaDetectAI)
![GitHub Discussions](https://img.shields.io/github/discussions/RanitManik/DermaDetectAI)
![GitHub License](https://img.shields.io/github/license/RanitManik/DermaDetectAI)
![wakatime](https://wakatime.com/badge/github/RanitManik/DermaDetectAI.svg)

DermaDetectAI is a Flask-based application developed to detect various skin diseases using deep learning models. This project was created as part of a college initiative by **Ranit Kumar Manik**, **Mohammad**, **Sayak Bal**, and **Partha Sarathi Manna**. It features three distinct models, each trained on different datasets using PyTorch to identify 5, 10, and 23 skin diseases, respectively.

## Table of Contents

- [Pre-trained Models](#pre-trained-models)
- [Setup Instructions](#setup-instructions)
    - [Install Dependencies](#install-dependencies)
    - [Running the Application](#running-the-application)
- [Using the Pre-trained Models](#using-the-pre-trained-models)
- [Training the Models](#training-the-models)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Pre-trained Models

The repository includes pre-trained models for skin disease detection:

1. **Model 1**: Detects 5 diseases. Trained on a ~69MB dataset with 98% validation accuracy.
2. **Model 2**: Detects 10 diseases. Trained on a ~2GB dataset with 85% validation accuracy.
3. **Model 3**: Detects 23 diseases. Trained on a ~6GB dataset with 45% validation accuracy.

## Setup Instructions

### Install Dependencies

Each model has its own `requirements.txt` file. To install the dependencies for a specific model, navigate to the respective model directory and run:

```bash
pip install -r requirements.txt
```

### Running the Application

To start the Flask application for a specific model, navigate to its directory and execute:

```bash
python app.py
```

The Flask server will start, and you can access the application at `http://127.0.0.1:5000`. Use the web interface to upload an image and receive disease predictions.

## Using the Pre-trained Models

The pre-trained models are included in the repository, allowing you to use them directly without additional training.

## Training the Models

To train the models from scratch, navigate to the `src` directory of the respective model and run `main.py`. Ensure that you have the dataset in the appropriate directory and adjust the `num_classes` parameter according to your dataset's number of classes.

```bash
python src/main.py
```

> [!NOTE]
>  This project is configured to utilize NVIDIA GPUs for faster training and inference. Make sure you have the necessary NVIDIA drivers, CUDA toolkit, and the GPU version of PyTorch installed. <br/>
> For GPU setup instructions, refer to the [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide). For PyTorch installation guidance, visit the [PyTorch Installation Page](https://pytorch.org/get-started/locally/).

## Project Structure

Here’s an overview of the project structure:

```
DermaDetectAI/
├── LICENSE
├── README.md
├── model-X/
│   ├── app.py
│   ├── models/
│   │   └── skin_disease_model.pth
│   ├── requirements.txt
│   ├── src/
│   │   └── main.py
│   ├── templates/
│   │   ├── result.html
│   │   └── upload.html
│   └── uploads/
│       └── [user_uploaded_files]
└── [other_files_and_directories]
```

For more details, refer to the [Project Structure Documentation](docs/project%20structure.md).

## Contributing

We welcome contributions to this project! To contribute, please follow these steps:

1. **Fork the repository**: Click the "Fork" button at the top right of this page to create a copy of the repository under your GitHub account.
2. **Clone the repository**: Clone your forked repository to your local machine:
    ```bash
    git clone https://github.com/RanitManik/DermaDetectAI.git
    ```
3. **Create a new branch**: Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature-or-bugfix-name
    ```
4. **Make your changes**: Implement your changes to the codebase.
5. **Commit your changes**: Commit your changes with a descriptive message:
    ```bash
    git commit -m "Description of your changes"
    ```
6. **Push to your branch**: Push your changes to your forked repository:
    ```bash
    git push origin feature-or-bugfix-name
    ```
7. **Create a Pull Request**: Open a pull request from your forked repository’s branch to the `main` branch of the original repository.

For detailed contribution guidelines, please refer to the [CONTRIBUTING.md](.github/CONTRIBUTING.md) file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
