import numpy as np

class WealthMetrics:
    @staticmethod
    def calculate_gini(balances: dict) -> float:
        """
        Calculates the Gini coefficient for the distribution of tokens.
        A Gini coefficient of 0 expresses perfect equality, 1 expresses maximal inequality.
        """
        values = np.array(list(balances.values()), dtype=float)
        if len(values) == 0:
            return 0.0
        
        # Sort the values
        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return ((np.sum((2 * index - n - 1) * values)) / (n * np.sum(values)))

    @staticmethod
    def get_summary(balances: dict) -> dict:
        """Returns a summary of wealth distribution."""
        values = list(balances.values())
        return {
            "total_wealth": sum(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std_dev": np.std(values),
            "max": max(values),
            "min": min(values),
            "gini": WealthMetrics.calculate_gini(balances)
        }
