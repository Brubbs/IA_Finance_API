# IA_Finance_API

A Data Mining project focused on financial market analysis using Machine Learning techniques to explore historical stock market data, identify patterns, and evaluate stock price movements.

## Objective

Develop a classification model based on the **Random Forest** algorithm to analyze historical stock market data, explore technical indicators, and evaluate the model's ability to identify patterns in stock price movements.

## Technologies

- **Python** – Main programming language.
- **Pandas** – Data manipulation and analysis.
- **NumPy** – Numerical operations.
- **Scikit-learn** – Machine Learning algorithms and model evaluation.
- **Requests** – API communication.
- **yfinance / brapi** – Financial market data retrieval.

## Project Structure

```text
Finance_RandomForest/
│
├── Finance.py
├── finance_data.py
├── finance_features.py
├── finance_model.py
├── .env
│
├── data/
│   └── historico/
│
└── README.md
```

## Features

- Historical stock market data collection.
- Financial data processing and organization.
- Feature engineering.
- Extraction of technical indicators for price analysis.
- Random Forest classification model training.
- Model performance evaluation.

## Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd Finance_RandomForest
```

Install the dependencies:

```bash
pip install pandas numpy scikit-learn requests yfinance python-dotenv
```

## Configuration

If using the brapi API, create a `.env` file in the project root directory:

```env
BRAPI_API_KEY=your_api_token
```

You can obtain an API token from the [brapi website](https://brapi.dev/).

Keep your credentials private and do not upload the `.env` file to GitHub.

## Usage

Run the main script:

```bash
python Finance.py
```

## Machine Learning Model

The project uses the **Random Forest** algorithm, an ensemble learning method based on multiple decision trees, to perform classification tasks using features extracted from financial data.

Different model hyperparameters can be explored, including:

- `n_estimators`: Number of trees in the forest.
- `max_depth`: Maximum depth of each tree.
- `min_samples_split`: Minimum number of samples required to split an internal node.

Model evaluation allows different configurations to be compared and their performance analyzed.

## Notes

- Data availability and historical coverage depend on the selected data provider and its API limitations.
- Model results depend on data quality, feature selection, and training configurations.
- This project is intended for educational and experimental purposes and does not constitute investment advice.

## License

This project was developed for academic and learning purposes in Data Mining and Machine Learning.
