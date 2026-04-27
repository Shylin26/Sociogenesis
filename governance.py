from economy import ComputeEconomy

class Governance:
    """
    Manages social rules and economic interventions for the Sociogenesis civilization.
    """
    def __init__(self, economy: ComputeEconomy):
        self.economy = economy
        self.tax_rate = 0.05  # 5% flat tax per period
        self.treasury = 0.0

    def collect_taxes(self) -> float:
        """
        Collects a flat tax from all agents and adds it to the treasury.
        Returns the amount collected.
        """
        total_collected = 0.0
        for agent_id, balance in self.economy.balances.items():
            tax_amount = int(balance * self.tax_rate)
            if tax_amount > 0:
                self.economy.balances[agent_id] -= tax_amount
                total_collected += tax_amount
        
        self.treasury += total_collected
        return total_collected

    def redistribute_wealth(self, strategy: str = "equal"):
        """
        Redistributes the treasury back to agents.
        'equal' strategy gives every agent an equal share.
        """
        if self.treasury <= 0:
            return

        n_agents = len(self.economy.balances)
        if strategy == "equal":
            share = int(self.treasury / n_agents)
            if share > 0:
                for agent_id in self.economy.balances:
                    self.economy.balances[agent_id] += share
                self.treasury -= (share * n_agents)
        
    def get_treasury_status(self) -> float:
        return self.treasury
