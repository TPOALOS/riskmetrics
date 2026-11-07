"""Reading return or price series from CSV files."""
import csv
import math

FREQUENCIES = {"daily": 252, "daily365": 365, "weekly": 52, "monthly": 12, "quarterly": 4, "yearly": 1}
MISSING = {"", "nan", "null", "none", "na", "n/a", "-", "--"}
DATE_NAMES = {"date", "time", "timestamp", "datetime", "period", "day", "month"}


class DataError(ValueError):
    pass


class Dataset:
    def __init__(self):
        self.series, self.labels, self.warnings = {}, {}, []


def parse_number(text):
    """float, or None for missing markers. A trailing % divides by 100. Raises ValueError for anything else."""
    t = text.strip().replace(" ", "")
    if t.lower() in MISSING:
        return None
    pct = t.endswith("%")
    v = float(t[:-1] if pct else t)
    if math.isnan(v) or math.isinf(v):
        return None
    return v / 100.0 if pct else v


def is_number(text):
    try:
        parse_number(text)
        return text.strip().lower() not in MISSING
    except ValueError:
        return False


def read_rows(path):
    try:
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            delim = max(",;\t", key=sample.count) if any(d in sample for d in ",;\t") else ","
            return [[c for c in row] for row in csv.reader(fh, delimiter=delim) if any(c.strip() for c in row)]
    except OSError as exc:
        raise DataError("cannot read %s: %s" % (path, exc)) from exc


def load_csv(path, columns=None, prices=False, date_column=None):
    """Load numeric columns as per-period returns. `columns`: names (case-insensitive) or 1-based indexes."""
    rows = read_rows(path)
    if len(rows) < 2:
        raise DataError("%s has no data rows" % path)
    width = max(len(r) for r in rows)
    has_header = any(c.strip() and not is_number(c) and c.strip().lower() not in MISSING for c in rows[0])
    header = [c.strip() or "col%d" % (i + 1) for i, c in enumerate(rows[0])] if has_header else ["col%d" % (i + 1) for i in range(width)]
    header += ["col%d" % (i + 1) for i in range(len(header), width)]
    body = rows[1:] if has_header else rows

    cells = lambda j: [(r[j] if j < len(r) else "") for r in body]
    date_idx = None
    if date_column is not None:
        if date_column.lower() not in [h.lower() for h in header]:
            raise DataError("date column %r not found (columns: %s)" % (date_column, ", ".join(header)))
        date_idx = [h.lower() for h in header].index(date_column.lower())
    else:
        for j, h in enumerate(header):
            if h.lower() in DATE_NAMES or (j == 0 and has_header and not all(is_number(c) or c.strip().lower() in MISSING for c in cells(0))):
                date_idx = j
                break
    ds = Dataset()
    numeric = {}
    for j, name in enumerate(header):
        if j == date_idx:
            continue
        col = cells(j)
        try:
            vals = [parse_number(c) for c in col]
        except ValueError:
            ds.warnings.append("skipped non-numeric column %r" % name)
            continue
        if all(v is None for v in vals):
            ds.warnings.append("skipped empty column %r" % name)
            continue
        numeric[name] = vals
    if columns:
        chosen = {}
        lower = {n.lower(): n for n in numeric}
        for c in columns:
            if c.lower() in lower:
                chosen[lower[c.lower()]] = numeric[lower[c.lower()]]
            elif c.isdigit() and 1 <= int(c) <= len(header) and header[int(c) - 1] in numeric:
                chosen[header[int(c) - 1]] = numeric[header[int(c) - 1]]
            else:
                raise DataError("column %r not found among numeric columns: %s" % (c, ", ".join(numeric) or "none"))
        numeric = chosen
    if not numeric:
        raise DataError("no numeric columns found in %s" % path)
    labels = [c.strip() for c in cells(date_idx)] if date_idx is not None else None
    for name, vals in numeric.items():
        kept = [(v, labels[i] if labels else None) for i, v in enumerate(vals) if v is not None]
        dropped = len(vals) - len(kept)
        if dropped:
            ds.warnings.append("%s: ignored %d empty/invalid value(s)" % (name, dropped))
        series = [v for v, _l in kept]
        lab = [l for _v, l in kept] if labels else None
        if prices:
            if any(p <= 0 for p in series):
                raise DataError("%s: prices must be positive" % name)
            series = [series[i] / series[i - 1] - 1.0 for i in range(1, len(series))]
            lab = lab[1:] if lab else None
        elif series and all(v > 0 for v in series) and sum(series) / len(series) > 1.0:
            ds.warnings.append("%s: all values are positive and large; if these are prices, use --prices" % name)
        if any(v <= -1.0 for v in series):
            raise DataError("%s: contains a return of -100%% or worse; are these prices? use --prices" % name)
        if len(series) < 2:
            raise DataError("%s: needs at least 2 observations" % name)
        ds.series[name], ds.labels[name] = series, lab
    return ds
