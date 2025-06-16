import os.path

from pydantic import BaseModel, Field
import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser

import csv
import pandas as pd
import numpy as np
import os


class CSVAnalysisTool:
    """
    Comprehensive CSV Analysis Tool for Agent Binod
    Handles CSV loading, data structure creation, and analysis
    """

    def __init__(self, file_path=None):
        self.data = None
        self.file_path = file_path
        self.columns = []
        self.data_types = {}
        self.summary_stats = {}
        self.analysis_results = {}

        if self.file_path:
            self.load_csv()

    def load_csv(self,  file_path=None) -> str:
        """
        Load CSV file and create data structure
        """
        if not file_path:
            raise Exception("File path need to load csv")

        self.file_path = file_path
        try:
            if not os.path.exists(self.file_path):
                raise Exception(f"Error: File not found at {self.file_path}")

            delimiters = [',', ';', '\t', '|']
            encodings = ['utf-8', 'latin-1', 'cp1252']

            for delimiter in delimiters:
                for encoding in encodings:
                    try:
                        self.data = pd.read_csv(
                            self.file_path,
                            delimiter=delimiter,
                            encoding=encoding,
                            low_memory=False
                        )
                        if len(self.data.columns) > 1:  # Successfully parsed
                            self.file_path = self.file_path
                            self.columns = list(self.data.columns)
                            self._analyze_data_types()

                            return f"""**CSV Loaded Successfully!**
                                **File:** {os.path.basename(self.file_path)}
                                **Shape:** {self.data.shape[0]} rows × {self.data.shape[1]} columns
                                **Columns:** {', '.join(self.columns[:5])}{'...' if len(self.columns) > 5 else ''}
                                **Delimiter:** '{delimiter}' | **Encoding:** {encoding}
                                
                                **Data Preview:**
                                {self.data.head(3).to_string()}"""
                    except Exception:
                        continue

            return "Error: Could not parse CSV file. Please check the format."

        except Exception as e:
            return str(e)

    def _analyze_data_types(self):
        """Analyze and categorize data types"""
        self.data_types = {}

        for column in self.data.columns:
            dtype = str(self.data[column].dtype)

            if dtype in ['int64', 'int32', 'float64', 'float32']:
                self.data_types[column] = 'numeric'
            elif dtype == 'object':
                sample_values = self.data[column].dropna().head(10)

                try:
                    pd.to_datetime(sample_values.iloc[0])
                    self.data_types[column] = 'datetime'
                except:
                    # Check if categorical (limited unique values)
                    unique_ratio = len(self.data[column].unique()) / len(self.data[column])
                    if unique_ratio < 0.1:  # Less than 10% unique values
                        self.data_types[column] = 'categorical'
                    else:
                        self.data_types[column] = 'text'
            else:
                self.data_types[column] = 'other'

    def get_data_structure(self) -> str:
        """
        Get detailed data structure information
        """
        if self.data is None:
            raise Exception("No CSV data loaded. Use load_csv first.")

        structure_info = f"""**Data Structure Analysis**
            **Basic Info:**
            **Rows:** {self.data.shape[0]:,}
            **Columns:** {self.data.shape[1]}
            **Memory Usage:** {self.data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB

            **Column Analysis:**"""

        for i, column in enumerate(self.data.columns, 1):
            col_data = self.data[column]
            dtype = self.data_types.get(column, 'unknown')
            missing_count = col_data.isnull().sum()
            missing_pct = (missing_count / len(col_data)) * 100
            unique_count = col_data.nunique()

            structure_info += f"""
                {i}. **{column}**
                • Type: {dtype} ({col_data.dtype})
                • Missing: {missing_count} ({missing_pct:.1f}%)
                • Unique: {unique_count:,}"""

            if dtype in ['categorical', 'text'] and unique_count <= 10:
                sample_values = col_data.value_counts().head(3).index.tolist()
                structure_info += f"""• Sample values: {', '.join(map(str, sample_values))}"""

        return structure_info

    def analyze_data(self, analysis_type: str = "general") -> str:
        """
        Perform comprehensive data analysis
        """
        if self.data is None:
            raise Exception("No CSV data loaded. Use load_csv first.")

        if analysis_type.lower() == "general":
            return self._general_analysis()
        elif analysis_type.lower() == "numeric":
            return self._numeric_analysis()
        elif analysis_type.lower() == "categorical":
            return self._categorical_analysis()
        elif analysis_type.lower() == "missing":
            return self._missing_data_analysis()
        elif analysis_type.lower() == "correlation":
            return self._correlation_analysis()
        else:
            return self._general_analysis()

    def _general_analysis(self) -> str:
        """General overview analysis"""
        numeric_cols = [col for col, dtype in self.data_types.items() if dtype == 'numeric']
        categorical_cols = [col for col, dtype in self.data_types.items() if dtype == 'categorical']
        text_cols = [col for col, dtype in self.data_types.items() if dtype == 'text']

        analysis = f"""**General Data Analysis**

            **Dataset Overview:**
            • **Total Records:** {len(self.data):,}
            • **Numeric Columns:** {len(numeric_cols)}
            • **Categorical Columns:** {len(categorical_cols)}
            • **Text Columns:** {len(text_cols)}
            • **Missing Data:** {self.data.isnull().sum().sum():,} cells

            **Data Quality:**
            • **Complete Rows:** {len(self.data.dropna()):,} ({(len(self.data.dropna()) / len(self.data) * 100):.1f}%)
            • **Duplicate Rows:** {self.data.duplicated().sum():,}"""

        # Top insights
        insights = []

        # Check for high missing data columns
        high_missing = self.data.isnull().sum()
        high_missing = high_missing[high_missing > len(self.data) * 0.5]
        if len(high_missing) > 0:
            insights.append(f"{len(high_missing)} columns have >50% missing data")

        # Check for constant columns
        constant_cols = [col for col in self.data.columns if self.data[col].nunique() <= 1]
        if constant_cols:
            insights.append(f"{len(constant_cols)} columns have constant values")

        # Check for high cardinality
        high_card_cols = [col for col in categorical_cols if
                          self.data[col].nunique() > len(self.data) * 0.5]
        if high_card_cols:
            insights.append(
                f"{len(high_card_cols)} categorical columns have very high cardinality")

        if insights:
            analysis += f"""
            **Key Insights:**
            {chr(10).join(f"• {insight}" for insight in insights)}"""

            return analysis

    def _numeric_analysis(self) -> str:
        """Analyze numeric columns"""
        numeric_cols = [col for col, dtype in self.data_types.items() if dtype == 'numeric']

        if not numeric_cols:
            return "No numeric columns found in the dataset."

        analysis = f"""**Numeric Data Analysis**
            **Numeric Columns:** {len(numeric_cols)}
            {', '.join(numeric_cols)}
            **Statistical Summary:**"""

        for col in numeric_cols[:5]:
            col_data = self.data[col].dropna()

            analysis += f"""
            **{col}:**
            • Mean: {col_data.mean():.2f}
            • Median: {col_data.median():.2f}
            • Std Dev: {col_data.std():.2f}
            • Min: {col_data.min():.2f}
            • Max: {col_data.max():.2f}
            • Range: {col_data.max() - col_data.min():.2f}"""

            # Check for outliers (values beyond 3 standard deviations)
            outliers = col_data[np.abs((col_data - col_data.mean()) / col_data.std()) > 3]
            if len(outliers) > 0:
                analysis += f"""Outliers: {len(outliers)} ({(len(outliers) / len(col_data) * 100):.1f}%)"""

        return analysis

    def _categorical_analysis(self) -> str:
        """Analyze categorical columns"""
        categorical_cols = [col for col, dtype in self.data_types.items() if dtype == 'categorical']

        if not categorical_cols:
            return "No categorical columns found in the dataset."

        analysis = f"""**Categorical Data Analysis**

            **Categorical Columns:** {len(categorical_cols)}"""

        for col in categorical_cols[:5]:  # Limit to first 5 columns
            value_counts = self.data[col].value_counts()

            analysis += f"""
            **{col}:**
            • Unique Values: {len(value_counts)}
            • Most Common: {value_counts.index[0]} ({value_counts.iloc[0]:,} occurrences)
            • Distribution:"""

            for i, (value, count) in enumerate(value_counts.head(5).items()):
                percentage = (count / len(self.data)) * 100
                analysis += f"""
                    {i + 1}. {value}: {count:,} ({percentage:.1f}%)"""

            if len(value_counts) > 5:
                analysis += f""".. and {len(value_counts) - 5} more values"""

        return analysis

    def _missing_data_analysis(self) -> str:
        """Analyze missing data patterns"""
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if len(missing_data) == 0:
            return "**No missing data found in the dataset!**"

        analysis = f"""**Missing Data Analysis**
            **Total Missing Values:** {missing_data.sum():,}
            **Columns with Missing Data:** {len(missing_data)}

            **Missing Data by Column:**"""

        for col, missing_count in missing_data.head(10).items():
            missing_pct = (missing_count / len(self.data)) * 100
            analysis += f"""**{col}:** {missing_count:,} ({missing_pct:.1f}%)"""

        analysis += f"""**Recommendations:**"""

        high_missing = missing_data[missing_data > len(self.data) * 0.5]
        if len(high_missing) > 0:
            analysis += f"""Consider dropping columns with >50% missing: {', '.join(high_missing.index)}"""

        medium_missing = missing_data[
            (missing_data > len(self.data) * 0.1) & (missing_data <= len(self.data) * 0.5)]
        if len(medium_missing) > 0:
            analysis += f"""Consider imputation for columns with 10-50% missing: {', '.join(medium_missing.index)}"""

        return analysis

    def _correlation_analysis(self) -> str:
        """Analyze correlations between numeric columns"""
        numeric_cols = [col for col, dtype in self.data_types.items() if dtype == 'numeric']

        if len(numeric_cols) < 2:
            return "Need at least 2 numeric columns for correlation analysis."

        correlation_matrix = self.data[numeric_cols].corr()

        analysis = f"""**Correlation Analysis**
            **Numeric Columns Analyzed:** {len(numeric_cols)}
            **Strong Correlations (|r| > 0.7):**"""

        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    strong_correlations.append((col1, col2, corr_value))

        if strong_correlations:
            for col1, col2, corr in sorted(strong_correlations, key=lambda x: abs(x[2]),
                                           reverse=True):
                direction = "positive" if corr > 0 else "negative"
                analysis += f"""**{col1}** ↔ **{col2}**: {corr:.3f} ({direction})"""
        else:
            analysis += """No strong correlations found (|r| > 0.7)"""

        # Add moderate correlations
        moderate_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if 0.3 < abs(corr_value) <= 0.7:
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    moderate_correlations.append((col1, col2, corr_value))

        if moderate_correlations:
            analysis += f"""**Moderate Correlations (0.3 < |r| ≤ 0.7):**"""
            for col1, col2, corr in sorted(moderate_correlations, key=lambda x: abs(x[2]),
                                           reverse=True)[:5]:
                direction = "positive" if corr > 0 else "negative"
                analysis += f"""**{col1}** ↔ **{col2}**: {corr:.3f} ({direction})"""

        return analysis

    def get_column_info(self, column_name: str) -> str:
        """Get detailed information about a specific column"""
        if self.data is None:
            return "No CSV data loaded. Use load_csv first."

        if column_name not in self.data.columns:
            available_cols = ', '.join(self.data.columns[:5])
            return f"Column '{column_name}' not found. Available columns: {available_cols}..."

        col_data = self.data[column_name]
        dtype = self.data_types.get(column_name, 'unknown')

        info = f"""**Column Analysis: {column_name}**

            **Basic Info:**
            • **Data Type:** {dtype} ({col_data.dtype})
            • **Total Values:** {len(col_data):,}
            • **Missing Values:** {col_data.isnull().sum():,} ({(col_data.isnull().sum() / len(col_data) * 100):.1f}%)
            • **Unique Values:** {col_data.nunique():,}"""

        if dtype == 'numeric':
            col_clean = col_data.dropna()
            info += f"""
            **Statistical Summary:**
            • **Mean:** {col_clean.mean():.2f}
            • **Median:** {col_clean.median():.2f}
            • **Mode:** {col_clean.mode().iloc[0] if len(col_clean.mode()) > 0 else 'N/A'}
            • **Standard Deviation:** {col_clean.std():.2f}
            • **Min:** {col_clean.min():.2f}
            • **Max:** {col_clean.max():.2f}
            • **Range:** {col_clean.max() - col_clean.min():.2f}"""

        elif dtype in ['categorical', 'text']:
            value_counts = col_data.value_counts()
            info += f"""**Value Distribution:**"""
            for i, (value, count) in enumerate(value_counts.head(10).items()):
                percentage = (count / len(col_data)) * 100
                info += f"""**{value}:** {count:,} ({percentage:.1f}%)"""

            if len(value_counts) > 10:
                info += f"""... and {len(value_counts) - 10} more unique values"""

        return info


class CalculatorTool:
    """Custom calculator tool"""

    def __init__(self):
        self.name = "Calculator"
        self.description = "Perform basic mathematical calculations"

    def run(self, expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            # Only allow safe mathematical operations
            expression = expression.strip()
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"

            if any(dangerous in expression for dangerous in ['__', 'import', 'exec', 'eval']):
                return "Error: Invalid expression"

            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"


class ParsedResponse(BaseModel):
    """Structured response model for output parsing"""
    response: str = Field(description="The main response text")
    word_count: int = Field(description="Number of words in response")
    timestamp: str = Field(description="When the response was generated")
    category: str = Field(description="Category of the response")

class CustomOutputParser(PydanticOutputParser):
    """Custom output parser to demonstrate LangChain parsing capabilities"""

    def __init__(self):
        super().__init__(pydantic_object=ParsedResponse)