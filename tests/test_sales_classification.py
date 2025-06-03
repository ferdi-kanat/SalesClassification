import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sales_classification import specificity_score
from sklearn.metrics import confusion_matrix

class TestSalesClassification(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        self.y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

    def test_specificity_score(self):
        """Test the specificity score calculation."""
        expected_specificity = 0.75  # Manually calculated: TN/(TN+FP) = 3/(3+1)
        calculated_specificity = specificity_score(self.y_true, self.y_pred)
        self.assertAlmostEqual(calculated_specificity, expected_specificity)

    def test_data_loading(self):
        """Test if the data file can be loaded and has the expected columns."""
        try:
            df = pd.read_csv('../New_1000_Sales_Records.csv')
            required_columns = [
                'Region', 'Item Type', 'Sales Channel', 'Order Priority',
                'Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue',
                'Total Cost', 'Total Profit', 'Order year', 'Order Month',
                'Order Weekday', 'Unit Margin', 'Order_Ship_Days'
            ]
            for col in required_columns:
                self.assertIn(col, df.columns)
        except FileNotFoundError:
            self.fail("Data file not found")

if __name__ == '__main__':
    unittest.main()
