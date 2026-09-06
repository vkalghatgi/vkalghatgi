"""Long-Pelosi / Short-Cramer market-neutral backtest.

Point-in-time correct: every signal is time-stamped with the date it became
*publicly available* (House Clerk filing date for Pelosi, episode air date for
Cramer), and the earliest trade happens on the next trading session.
"""

__version__ = "0.1.0"
