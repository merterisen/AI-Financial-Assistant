from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict

class IncomeStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revenue: Optional[float] = Field(None, description="Total revenue for the most recent period in the table.")
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    other_line_items: Dict[str, Optional[float]] = Field(default_factory=dict)



class BalanceSheet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    other_line_items: Dict[str, Optional[float]] = Field(default_factory=dict)



class CashFlow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    net_cash_from_operating_activities: Optional[float] = None
    net_cash_from_investing_activities: Optional[float] = None
    net_cash_from_financing_activities: Optional[float] = None
    other_line_items: Dict[str, Optional[float]] = Field(default_factory=dict)



class FinancialStatements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    income_statement: IncomeStatement
    balance_sheet: BalanceSheet
    cash_flow: CashFlow