"""Download and parse House of Representatives Periodic Transaction Reports (PTRs).

Source: https://disclosures-clerk.house.gov (Office of the Clerk, Financial Disclosure)

The yearly ``{year}FD.zip`` archives contain a tab-separated index with one row per
filing (filer, filing type, *FilingDate*, DocID). Filing type ``P`` is a PTR.
PTR PDFs live at ``public_disc/ptr-pdfs/{year}/{DocID}.pdf`` and, for electronically
filed reports (all of Pelosi's since 2015), contain machine-readable text.

Point-in-time note
------------------
The STOCK Act requires a PTR to be filed within 30 days of the member being
*notified* of a trade and no later than 45 days after the trade itself. The
information only becomes public when the Clerk posts the PDF, i.e. on/after the
``FilingDate`` in the index. We therefore carry **both** the transaction date and
the filing date, and the backtest only acts on the latter.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://disclosures-clerk.house.gov/public_disc"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; pelosi-cramer-neutral backtest)"}

FD_COLUMNS = ["Prefix", "Last", "First", "Suffix", "FilingType", "StateDst", "Year", "FilingDate", "DocID"]


# --------------------------------------------------------------------------- index
def download_fd_index(year: int, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    txt_path = cache_dir / f"{year}FD.txt"
    if not txt_path.exists():
        r = requests.get(f"{BASE}/financial-pdfs/{year}FD.zip", headers=HEADERS, timeout=60)
        if r.status_code != 200 or not r.content:
            return pd.DataFrame(columns=FD_COLUMNS)
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
                txt_path.write_bytes(z.read(name))
        except (zipfile.BadZipFile, StopIteration):
            return pd.DataFrame(columns=FD_COLUMNS)
    df = pd.read_csv(txt_path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def list_ptr_filings(last: str, first: str, years: range, cache_dir: Path) -> pd.DataFrame:
    frames = []
    for y in years:
        idx = download_fd_index(y, cache_dir)
        if idx.empty:
            continue
        m = (idx["Last"].str.lower() == last.lower()) & idx["First"].str.lower().str.startswith(first.lower())
        frames.append(idx[m & (idx["FilingType"] == "P")])
    if not frames:
        return pd.DataFrame(columns=FD_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["FilingDate"] = pd.to_datetime(out["FilingDate"], format="%m/%d/%Y")
    out["Year"] = out["Year"].astype(int)
    return out.sort_values(["FilingDate", "DocID"]).reset_index(drop=True)


def download_ptr_pdf(year: int, doc_id: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{doc_id}.pdf"
    if not p.exists():
        r = requests.get(f"{BASE}/ptr-pdfs/{year}/{doc_id}.pdf", headers=HEADERS, timeout=60)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p


# --------------------------------------------------------------------------- parsing
_DATE = r"\d{1,2}/\d{1,2}/\d{4}"
# Transaction header: "<type> [(partial)] <txn date> <notif date> $<lower> -"
_HEADER_RE = re.compile(
    rf"\s([PSEpse])(?:\s*\((?:partial|Partial)\))?\s+({_DATE})\s+({_DATE})\s+\$([\d,\.]+)(\s*-)?",
)
_TICKER_RE = re.compile(r"\(([A-Za-z][A-Za-z0-9\.\-]{0,7})\)")
_ASSET_TYPE_RE = re.compile(r"\[([A-Za-z]{2})\]")
_UPPER_AMT_RE = re.compile(r"\$([\d,]+)")
_STATUS_RE = re.compile(r"^F[A-Za-z ]*:\s*(New|Amended|Deleted)", re.I)
_DESC_RE = re.compile(r"^D[A-Za-z ]*:\s*(.*)")
_NOISE_RE = re.compile(r"^(ID Owner Asset|Type Date Gains|\$200\?|\* For the complete list)", re.I)
_GLYPH_RE = re.compile(r"\b(gfedcb?|nmlkji?)\b")
# Form boilerplate that follows the transaction table (case is often garbled by the PDF font)
_FOOTER_RE = re.compile(
    r"^(?:[A-Za-z]\s{2,}\S.*|initial public offerings?|comments|certification.*|yes\s+no|hon\..*|"
    r"i certify.*|digitally signed.*|\*.*|location:.*|best of my knowledge.*|my knowledge and belief.*)$",
    re.I,
)

# Words that appear inside parentheses but are not tickers
_NOT_TICKERS = {"partial", "one", "two", "three", "four", "five", "shares", "share"}

# Assets whose PTR line has no ticker in parentheses but that are exchange listed
_NAME_TO_TICKER = {
    "roblox corporation": "RBLX",
    "alliancebernstein holding": "AB",
    "alliance bernstein holding": "AB",
}


def _ticker_from_name(name: str) -> str | None:
    n = name.lower()
    for k, v in _NAME_TO_TICKER.items():
        if k in n:
            return v
    return None


def _amount_bounds(lower: str, upper: str | None) -> tuple[float, float]:
    lo = float(lower.replace(",", ""))
    hi = float(upper.replace(",", "")) if upper else lo
    return lo, hi


def parse_ptr_text(text: str, doc_id: str, filing_date: pd.Timestamp) -> list[dict]:
    """Parse one PTR's extracted text into transaction records."""
    lines = [_GLYPH_RE.sub("", ln).rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip() and not _NOISE_RE.match(ln.strip())]

    records: list[dict] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = _HEADER_RE.search(ln)
        if not m or _STATUS_RE.match(ln) or _DESC_RE.match(ln):
            i += 1
            continue

        head_pre = ln[: m.start()]            # owner + asset name (+ maybe ticker/type)
        head_post = ln[m.end():]              # maybe upper amount
        block_extra: list[str] = []
        status, desc = None, []
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if _HEADER_RE.search(nxt) and not _DESC_RE.match(nxt):
                break
            s = nxt.strip()
            sm = _STATUS_RE.match(s)
            dm = _DESC_RE.match(s)
            if not (sm or dm) and _FOOTER_RE.match(s):
                if s.lower().startswith("location:"):
                    j += 1
                    continue
                break
            if sm:
                status = sm.group(1).title()
            elif dm:
                desc.append(dm.group(1))
            elif desc:                        # continuation of description
                desc.append(s)
            elif status is None:
                block_extra.append(nxt)
            j += 1

        asset_text = " ".join([head_pre] + block_extra)
        tickers = [t for t in _TICKER_RE.findall(asset_text) if t.lower() not in _NOT_TICKERS]
        ticker = tickers[-1].upper() if tickers else _ticker_from_name(asset_text)
        at = _ASSET_TYPE_RE.search(asset_text + " " + head_post)
        asset_type = at.group(1).upper() if at else None
        upper = _UPPER_AMT_RE.search(" ".join([head_post] + block_extra))
        lo, hi = _amount_bounds(m.group(4), upper.group(1) if upper else None)

        owner_m = re.match(r"^\s*(?:\d{10}\s+)?(SP|JT|DC)\b", head_pre, re.I)
        name = re.sub(r"^\s*(?:\d{10}\s+)?(SP|JT|DC)\s+", "", head_pre, flags=re.I)
        name = _TICKER_RE.sub("", _ASSET_TYPE_RE.sub("", name)).strip(" -")

        records.append(
            dict(
                doc_id=doc_id,
                filing_date=filing_date.normalize(),
                owner=owner_m.group(1).upper() if owner_m else "",
                asset=name,
                ticker=ticker,
                asset_type=asset_type,
                txn_type=m.group(1).upper(),
                partial="partial" in ln[m.start(): m.end()].lower(),
                txn_date=pd.to_datetime(m.group(2), format="%m/%d/%Y"),
                notif_date=pd.to_datetime(m.group(3), format="%m/%d/%Y"),
                amount_low=lo,
                amount_high=hi,
                status=status or "New",
                description=" ".join(desc).strip(),
            )
        )
        i = j
    return records


def extract_pdf_text(pdf_path: Path) -> str:
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join((pg.extract_text() or "") for pg in pdf.pages)


def classify_instrument(row: pd.Series) -> str:
    """Return 'stock', 'call', 'put', 'exercise', 'other' from asset-type code + description."""
    d = (row.get("description") or "").lower()
    at = row.get("asset_type")
    if "exercis" in d:
        return "exercise"
    if at == "OP" or "call option" in d or "put option" in d or re.search(r"\b\d+\s+options?\b", d):
        return "put" if "put" in d else "call"
    if at in (None, "ST", "AB") or "shares" in d:
        return "stock"
    return "other"


def build_pelosi_transactions(
    cache_dir: Path,
    years: range = range(2012, 2027),
    last: str = "Pelosi",
    first: str = "Nancy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download index + PDFs and return (filings_index, transactions)."""
    filings = list_ptr_filings(last, first, years, cache_dir / "fd_index")
    recs: list[dict] = []
    for _, f in filings.iterrows():
        pdf = download_ptr_pdf(int(f["Year"]), f["DocID"], cache_dir / "ptr_pdfs")
        text = extract_pdf_text(pdf)
        recs.extend(parse_ptr_text(text, f["DocID"], f["FilingDate"]))
    tx = pd.DataFrame(recs)
    if tx.empty:
        return filings, tx
    tx["instrument"] = tx.apply(classify_instrument, axis=1)
    tx["disclosure_lag_days"] = (tx["filing_date"] - tx["txn_date"]).dt.days
    # Amendments re-state a transaction already reported (often with reworded
    # descriptions); keep the earliest public record so the signal date is the
    # first date the information was available.
    tx = tx.sort_values(["filing_date", "doc_id"]).reset_index(drop=True)
    key = ["ticker", "txn_type", "txn_date", "amount_low", "amount_high"]
    first_filing = tx.groupby(key, dropna=False)["filing_date"].transform("min")
    is_restated = (tx["status"] == "Amended") & (tx["filing_date"] > first_filing)
    tx = tx[~is_restated]
    return filings, tx.sort_values(["filing_date", "txn_date"]).reset_index(drop=True)
