import pandas as pd

from pcn.data.house_clerk import classify_instrument, parse_ptr_text

MODERN = """Filing ID #20033725
P        T           R
ID Owner Asset Transaction Date Notification Amount Cap.
Type Date Gains >
$200?
SP Alphabet Inc. - Class A Common P 01/16/2026 01/16/2026 $500,001 -
Stock (GOOGL) [ST] $1,000,000
F      S     : New
D          : Exercised 50 call options purchased 1/14/25 (5,000 shares) at a strike price of $150 with an expiration date of
1/16/26.
SP Amazon.com, Inc. - Common Stock S (partial) 12/24/2025 12/24/2025 $1,000,001 -
(AMZN) [ST] $5,000,000
F      S     : New
ID Owner Asset Transaction Date Notification Amount Cap.
Type Date Gains >
$200?
D          : Sold 20,000 shares.
SP NVIDIA Corporation - Common Stock P 12/30/2025 12/30/2025 $100,001 -
(NVDA) [OP] $250,000
F      S     : New
D          : Purchased 20 call options with a strike price of $100 and an expiration date of 1/15/27.
SP REOF XXV, LLC [AB] P 12/22/2025 12/22/2025 $500,001 - gfedc
$1,000,000
F      S     : New
D          : Investment in LLC.
* For the complete list of asset type abbreviations, please visit https://fd.house.gov/reference/asset-type-codes.aspx.
I       P      O
Yes No
C                 S
I CERTIFY that the statements I have made on the attached Periodic Transaction Report are true, complete, and correct to the best of
my knowledge and belief.
Digitally Signed: Hon. Nancy Pelosi , 01/23/2026
"""

LEGACY = """Filing ID #20002351
tranSactionS
iD owner asset transaction Date notification amount
type Date
Visa Inc. (V) s 12/29/2014 12/29/2014 $250,001 - $500,000
FIlINg sTaTus: New
DEsCRIPTION: sale of 1000 shares
Walt Disney Company (DIs) P 12/10/2014 12/10/2014 $50,001 - $100,000
FIlINg sTaTus: New
DEsCRIPTION: Purchase of 92 Options
commentS
initial Public offeringS
nmlkj Yes nmlkji No
certification anD Signature
"""


def test_modern_layout_parses_wrapped_tickers_and_page_breaks():
    recs = pd.DataFrame(parse_ptr_text(MODERN, "20033725", pd.Timestamp("2026-01-23")))
    assert len(recs) == 4
    by = recs.set_index("ticker", drop=False)
    assert by.loc["GOOGL", "txn_type"] == "P" and by.loc["GOOGL", "amount_high"] == 1_000_000
    assert by.loc["AMZN", "txn_type"] == "S" and bool(by.loc["AMZN", "partial"]) is True
    assert by.loc["AMZN", "description"] == "Sold 20,000 shares."          # survives the page-header interruption
    assert by.loc["NVDA", "asset_type"] == "OP"
    assert recs.ticker.isna().sum() == 1                                     # the private LLC has no ticker
    assert (recs.filing_date == pd.Timestamp("2026-01-23")).all()
    assert (recs.txn_date <= recs.filing_date).all()


def test_legacy_layout_normalises_case_and_strips_footer():
    recs = pd.DataFrame(parse_ptr_text(LEGACY, "20002351", pd.Timestamp("2015-01-12")))
    assert list(recs.ticker) == ["V", "DIS"]
    assert list(recs.txn_type) == ["S", "P"]
    assert recs.iloc[0].amount_low == 250_001 and recs.iloc[0].amount_high == 500_000
    assert "offering" not in recs.iloc[1].description.lower()
    assert recs.iloc[1].description == "Purchase of 92 Options"


def test_instrument_classification():
    assert classify_instrument(pd.Series(dict(asset_type="ST", description="Purchased 10,000 shares."))) == "stock"
    assert classify_instrument(pd.Series(dict(asset_type="OP", description="Purchased 20 call options ..."))) == "call"
    assert classify_instrument(pd.Series(dict(asset_type="ST", description="Exercised 50 call options (5,000 shares)"))) == "exercise"
    assert classify_instrument(pd.Series(dict(asset_type=None, description="Purchase of 92 Options"))) == "call"
