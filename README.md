# Sales Classification Analysis

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/stable/)

## 📋 Overview

This project implements an advanced machine learning pipeline for sales profit prediction using various classification algorithms. It features automated feature selection through genetic algorithms and dimensionality reduction using PCA, providing comprehensive performance analysis and visualizations.

## 🚀 Key Features

### Machine Learning Models

- **Classification Algorithms**
  - k-Nearest Neighbors (k-NN)
  - Naive Bayes
  - Decision Tree
  - Support Vector Machines (SVM)
    - Linear kernel
    - RBF kernel
    - Polynomial kernel
  - Logistic Regression
  - Random Forest

### Advanced Features

- **Genetic Algorithm Feature Selection**

  - Optimized feature subset selection
  - Automatic feature importance ranking
  - Cross-validation based evaluation

- **Principal Component Analysis (PCA)**
  - Automated dimensionality reduction
  - Variance analysis
  - Component importance visualization

### Performance Analysis

- Comprehensive metric evaluation
  - Accuracy, Precision, Recall
  - F1 Score
  - Matthews Correlation Coefficient
  - Confusion Matrices
- Detailed visualization suite
  - Decision tree structures
  - PCA variance plots
  - Performance comparison charts

## 📊 Sample Results

| Model         | Accuracy | Precision | Recall |
| ------------- | -------- | --------- | ------ |
| Decision Tree | 100%     | 99.8%     | 99.9%  |
| k-NN          | 98.5%    | 98.2%     | 98.7%  |
| SVM           | 99.5%    | 99.3%     | 99.4%  |
| Naive Bayes   | 63.5%    | 62.8%     | 63.9%  |

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/ferdi-kanat/SalesClassification.git
cd SalesClassification
```

2. **Set up Python environment**

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 📖 Usage

1. **Data**

   - The sample dataset `New_1000_Sales_Records.csv` is included in the repository
   - You can use your own data file following the same format (see Data Format section)

2. **Run the analysis**

```bash
python sales_classification.py
```

3. **Review outputs**
   - Check generated CSV files for detailed metrics
   - Examine visualization files for insights

## 📝 Data Format

The project includes a sample dataset (`New_1000_Sales_Records.csv`) that demonstrates the required format. You can use this as a template for your own data.

### Required Columns

```
Region            | Sales Channel    | Order Priority
Item Type        | Units Sold       | Unit Price
Unit Cost        | Total Revenue    | Total Cost
Total Profit     | Order year       | Order Month
Order Weekday    | Unit Margin      | Order_Ship_Days
```

## 📈 Output Files

### Performance Metrics

- `table1_classifier_performance.csv`: Standard classifier metrics
- `table2_ga_performance.csv`: Genetic algorithm results
- `table3_pca_performance.csv`: PCA analysis results

### Visualizations

- `decision_tree_visualization.png`: Tree structure analysis
- `*_confusion_matrix.png`: Per-classifier confusion matrices
- `pca_variance_ratios.png`: PCA component analysis
- `pca_cumulative_variance.png`: Cumulative variance trends

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to:

- Report bugs
- Suggest enhancements
- Submit pull requests
- Set up your development environment
- Follow our code style guidelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✍️ Author

- **Ferdi Kanat** - _Initial work_ - [GitHub Profile](https://github.com/ferdi-kanat)

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) documentation and community
- [DEAP framework](https://deap.readthedocs.io/) for genetic algorithms
- All contributors and maintainers

## 📧 Contact

For questions and feedback:

- Email: 200101038@ogrenci.yalova.edu.tr
- Project Link: https://github.com/ferdi-kanat/SalesClassification
