"""
features.py
-----------
Feature extraction for "Phishing URL Analytics and Classification".

Turns a raw URL string into numeric features of two kinds:
  1. Lexical features      - computed from the characters of the URL itself
  2. Domain-based features - computed from the host / domain part of the URL

No network calls are made (no WHOIS, no page download), so extraction is
fast enough to run on hundreds of thousands of URLs.

Used by:
  train.py - to build the training table from the dataset
  app.py   - to analyse a single URL typed in by the user

Quick test:  python features.py
"""

import math
import re
from collections import Counter
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd

# ---------------------------------------------------------------------------
# Reference lists
# ---------------------------------------------------------------------------

# Words that phishing pages commonly put in the URL to look trustworthy/urgent
SUSPICIOUS_WORDS = (
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "bank", "password", "webscr", "wallet", "payment", "billing", "suspend",
    "unlock", "alert", "bonus", "gift", "prize", "winner", "kyc", "otp",
    "recover", "validate", "authenticate",
)

# Well-known brands that attackers impersonate
BRANDS = (
    "paypal", "apple", "google", "microsoft", "amazon", "facebook", "netflix",
    "instagram", "whatsapp", "linkedin", "dropbox", "adobe", "outlook",
    "office365", "paytm", "phonepe", "hdfc", "icici", "flipkart",
)

# URL shortening services (they hide the real destination)
SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "rebrand.ly", "t.ly",
})

# Top-level domains that are cheap/free and frequently abused
ABUSED_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work", "click",
    "link", "buzz", "icu", "cam", "rest", "stream", "download", "loan",
    "win", "bid", "review", "date", "zip", "mov", "cyou", "cfd", "sbs",
    "monster", "quest",
})

# Long-established, widely used top-level domains
COMMON_TLDS = frozenset({
    "com", "org", "net", "edu", "gov", "in", "uk", "de", "fr", "jp", "au",
    "ca", "us", "co.in", "co.uk", "com.au", "co.jp", "gov.in", "ac.in",
    "org.uk", "ac.uk",
})

# Suffixes made of two labels, so "sbi.co.in" is read as domain "sbi"
TWO_LEVEL_SUFFIXES = frozenset({
    "co.in", "net.in", "org.in", "gov.in", "ac.in", "edu.in", "nic.in",
    "co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "co.jp", "com.br",
    "co.za", "com.cn", "com.mx", "co.nz", "com.sg", "co.kr", "com.tr",
})

# ---------------------------------------------------------------------------
# Feature name lists (the order here is the column order used everywhere)
# ---------------------------------------------------------------------------

LEXICAL_FEATURES = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equals",
    "num_at",
    "num_ampersands",
    "num_percent",
    "num_digits",
    "digit_ratio",
    "letter_ratio",
    "num_special_chars",
    "url_entropy",
    "path_length",
    "path_depth",
    "query_length",
    "num_query_params",
    "num_suspicious_words",
    "has_double_slash_in_path",
    "longest_word_length",
]

DOMAIN_FEATURES = [
    "is_https",
    "host_length",
    "domain_length",
    "num_subdomains",
    "has_ip_address",
    "has_port",
    "tld_length",
    "is_abused_tld",
    "is_common_tld",
    "domain_num_digits",
    "domain_num_hyphens",
    "domain_entropy",
    "is_shortener",
    "is_punycode",
    "brand_outside_domain",
]

FEATURE_NAMES = LEXICAL_FEATURES + DOMAIN_FEATURES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_SPECIAL_RE = re.compile(r"[^A-Za-z0-9]")
_WORD_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


def shannon_entropy(text: str) -> float:
    """Return the Shannon entropy (randomness) of a string, in bits per character."""
    if not text:
        return 0.0
    total = len(text)
    counts = Counter(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _is_ip(host: str) -> bool:
    """Return True if the host is an IP address rather than a domain name."""
    if not host:
        return False
    return bool(
        _IPV4_RE.match(host)       # 192.168.1.1
        or host.isdigit()          # 3232235777 (decimal form)
        or host.startswith("0x")   # 0xC0A80101 (hex form)
        or ":" in host             # IPv6
    )


def _parse(url: str) -> tuple[str, str, str, str, int | None]:
    """Split a URL into (clean_url, host, path, query, port). Never raises."""
    url = str(url).strip()
    to_parse = url if _SCHEME_RE.match(url) else "http://" + url
    try:
        parsed = urlparse(to_parse)
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
        try:
            port = parsed.port
        except ValueError:
            port = None
    except ValueError:
        host, path, query, port = "", "", "", None
    return url, host, path, query, port


def _split_host(host: str) -> tuple[str, str, str]:
    """Split a host into (subdomain, domain, tld).

    Example: "mail.sbi.co.in" -> ("mail", "sbi", "co.in")
    """
    if not host or _is_ip(host):
        return "", host, ""
    parts = host.split(".")
    if len(parts) == 1:
        return "", host, ""
    if len(parts) >= 3 and ".".join(parts[-2:]) in TWO_LEVEL_SUFFIXES:
        return ".".join(parts[:-3]), parts[-3], ".".join(parts[-2:])
    return ".".join(parts[:-2]), parts[-2], parts[-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tld(url: str) -> str:
    """Return the top-level domain of a URL, or "(ip)" / "(none)"."""
    _, host, _, _, _ = _parse(url)
    if _is_ip(host):
        return "(ip)"
    _, _, tld = _split_host(host)
    return tld or "(none)"


def extract_features(url: str) -> dict[str, float]:
    """Extract all lexical and domain-based features from one URL.

    Returns a dict whose keys are exactly FEATURE_NAMES.
    """
    url, host, path, query, port = _parse(url)
    lower = url.lower()
    length = len(url)

    digits = sum(ch.isdigit() for ch in url)
    letters = sum(ch.isalpha() for ch in url)
    words = [w for w in _WORD_SPLIT_RE.split(url) if w]

    subdomain, domain, tld = _split_host(host)
    sub_labels = [p for p in subdomain.split(".") if p and p != "www"]
    registered = f"{domain}.{tld}" if tld else domain
    outside_domain = subdomain + path.lower() + query.lower()

    features = {
        # ---- lexical ----
        "url_length": length,
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equals": url.count("="),
        "num_at": url.count("@"),
        "num_ampersands": url.count("&"),
        "num_percent": url.count("%"),
        "num_digits": digits,
        "digit_ratio": digits / length if length else 0.0,
        "letter_ratio": letters / length if length else 0.0,
        "num_special_chars": len(_SPECIAL_RE.findall(url)),
        "url_entropy": shannon_entropy(url),
        "path_length": len(path),
        "path_depth": len([seg for seg in path.split("/") if seg]),
        "query_length": len(query),
        "num_query_params": len([p for p in query.split("&") if p]),
        "num_suspicious_words": sum(word in lower for word in SUSPICIOUS_WORDS),
        "has_double_slash_in_path": int("//" in path),
        "longest_word_length": max((len(w) for w in words), default=0),
        # ---- domain-based ----
        "is_https": int(lower.startswith("https://")),
        "host_length": len(host),
        "domain_length": len(domain),
        "num_subdomains": len(sub_labels),
        "has_ip_address": int(_is_ip(host)),
        "has_port": int(port is not None and port not in (80, 443)),
        "tld_length": len(tld),
        "is_abused_tld": int(tld in ABUSED_TLDS),
        "is_common_tld": int(tld in COMMON_TLDS),
        "domain_num_digits": sum(ch.isdigit() for ch in host),
        "domain_num_hyphens": host.count("-"),
        "domain_entropy": shannon_entropy(host),
        "is_shortener": int(registered in SHORTENERS or host in SHORTENERS),
        "is_punycode": int("xn--" in host),
        "brand_outside_domain": int(
            domain not in BRANDS and any(b in outside_domain for b in BRANDS)
        ),
    }
    return features


def extract_features_df(urls: Iterable[str]) -> pd.DataFrame:
    """Extract features for many URLs and return them as a DataFrame."""
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def risk_flags(url: str) -> list[str]:
    """Return human-readable warning signs found in a URL (may be empty)."""
    f = extract_features(url)
    flags: list[str] = []
    if f["has_ip_address"]:
        flags.append("Uses a raw IP address instead of a domain name")
    if not f["is_https"]:
        flags.append("Does not use HTTPS")
    if f["num_at"]:
        flags.append("Contains '@', which can hide the real destination")
    if f["is_abused_tld"]:
        flags.append("Top-level domain is frequently abused for phishing")
    if f["is_shortener"]:
        flags.append("URL shortener hides the real destination")
    if f["is_punycode"]:
        flags.append("Punycode domain (possible lookalike characters)")
    if f["brand_outside_domain"]:
        flags.append("A brand name appears outside the main domain")
    if f["num_subdomains"] >= 3:
        flags.append(f"Many subdomains ({f['num_subdomains']})")
    if f["domain_num_hyphens"] >= 2:
        flags.append("Several hyphens in the domain")
    if f["num_suspicious_words"] >= 2:
        flags.append(f"Contains {f['num_suspicious_words']} sensitive keywords")
    if f["has_port"]:
        flags.append("Uses a non-standard port")
    if f["url_length"] > 75:
        flags.append(f"Unusually long URL ({f['url_length']} characters)")
    return flags


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "https://www.wikipedia.org",
        "http://192.168.10.5/paypal/login/verify.php?acc=123&session=9f8a7b",
    ]
    for sample in samples:
        print("\nURL:", sample)
        print("TLD:", get_tld(sample))
        for name, value in extract_features(sample).items():
            print(f"  {name:28s} {value}")
        print("Flags:", risk_flags(sample) or "none")
    print(f"\nTotal features: {len(FEATURE_NAMES)} "
          f"({len(LEXICAL_FEATURES)} lexical + {len(DOMAIN_FEATURES)} domain-based)")
