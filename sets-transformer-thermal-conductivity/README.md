# Sets Transformer for Thermal Conductivity Prediction

This project implements a Sets Transformer model to predict thermal conductivity based on elliptical structures. The model is trained using a dataset that includes both thermal conductivity matrices and corresponding elliptical features.

## Project Structure

- `configs/`: Contains configuration files for training.
- `data/`: Contains the dataset, including CSV files for thermal conductivity and JSON files for elliptical structures.
- `requirements.txt`: Lists the required Python libraries and dependencies.
- `scripts/`: Contains scripts for data preprocessing and training execution.
- `src/`: Contains the main source code, including dataset definitions, model implementations, training logic, and utility functions.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd sets-transformer-thermal-conductivity
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Ensure that the data files are placed in the `data/` directory. The CSV file should contain the thermal conductivity data, and the JSON files should contain the elliptical structure data.

2. **Training the Model**: You can start the training process by running the following command:

```bash
bash scripts/run_training.sh
```

This script will execute the training process using the configurations specified in `configs/training.yaml`.

3. **Evaluation**: After training, the model can be evaluated using the metrics defined in the `src/utils/metrics.py` file.

## Configuration

Training parameters and hyperparameters can be adjusted in the `configs/training.yaml` file. This includes settings such as learning rate, batch size, number of epochs, and early stopping criteria.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

This project is inspired by the need for efficient thermal conductivity prediction methods in materials science.