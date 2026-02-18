"""
LinkedIn Job Scraper — Streamlit page
Scrapes LinkedIn's public job search without requiring a login.
"""

import io
import re
import time
import random
from datetime import datetime
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LinkedIn Job Scraper",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────

LINKEDIN_JOBS_URL = "https://www.linkedin.com/jobs/search/"

JOB_TYPE_OPTIONS = {
    "Any": "",
    "Full-time": "F",
    "Part-time": "P",
    "Contract": "C",
    "Internship": "I",
    "Temporary": "T",
}

WORK_TYPE_OPTIONS = {
    "Any": "",
    "On-site": "1",
    "Remote": "2",
    "Hybrid": "3",
}

EXP_LEVEL_OPTIONS = {
    "Any": "",
    "Internship": "1",
    "Entry level": "2",
    "Associate": "3",
    "Mid-Senior level": "4",
    "Director": "5",
    "Executive": "6",
}

TIME_POSTED_OPTIONS = {
    "Any time": "",
    "Past 24 hours": "r86400",
    "Past week": "r604800",
    "Past month": "r2592000",
}

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# ── Scraping helpers ──────────────────────────────────────────────────────────


def _build_url(keyword: str, location: str, start: int,
               job_type: str, work_type: str,
               time_posted: str, exp_level: str) -> str:
    params: dict = {"keywords": keyword, "location": location, "start": start}
    if job_type:
        params["f_JT"] = job_type
    if work_type:
        params["f_WT"] = work_type
    if time_posted:
        params["f_TPR"] = time_posted
    if exp_level:
        params["f_E"] = exp_level
    return f"{LINKEDIN_JOBS_URL}?{urlencode(params)}"


def _parse_cards(soup: BeautifulSoup) -> list[dict]:
    """Extract job dicts from a parsed LinkedIn results page."""
    jobs: list[dict] = []

    # LinkedIn occasionally renames classes; try a cascade of selectors.
    cards = soup.find_all("div", class_="base-card")
    if not cards:
        cards = soup.find_all(
            "li",
            class_=lambda c: c and "jobs-search-results__list-item" in c,
        )
    if not cards:
        cards = soup.find_all(
            "div",
            class_=lambda c: c and "job-search-card" in c,
        )

    for card in cards:
        try:
            # Title
            title_el = card.find(
                ["h3", "h4"],
                class_=lambda c: c and "title" in c.lower(),
            ) or card.find("h3")

            # Company
            company_el = card.find(
                ["h4", "a"],
                class_=lambda c: c and (
                    "subtitle" in c.lower() or "company" in c.lower()
                ),
            ) or card.find("h4")

            # Location
            location_el = card.find(
                "span",
                class_=lambda c: c and "location" in c.lower(),
            )

            # Date
            time_el = card.find("time")

            # Apply link — prefer the canonical job-view URL
            link_el = card.find("a", href=lambda h: h and "/jobs/view/" in h)
            if not link_el:
                link_el = card.find(
                    "a", href=lambda h: h and "linkedin.com/jobs" in h
                )

            title = title_el.get_text(strip=True) if title_el else "N/A"
            company = company_el.get_text(strip=True) if company_el else "N/A"
            location = location_el.get_text(strip=True) if location_el else "N/A"

            if time_el:
                date_posted = time_el.get("datetime") or time_el.get_text(strip=True)
            else:
                date_posted = "N/A"

            if link_el:
                raw_href = link_el.get("href", "")
                # Strip tracking query params
                link = raw_href.split("?")[0]
            else:
                link = ""

            # Skip cards that produced nothing useful
            if title == "N/A" and company == "N/A":
                continue

            jobs.append(
                {
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Date Posted": date_posted,
                    "Apply Link": link,
                }
            )
        except Exception:
            continue

    return jobs


def scrape_linkedin_jobs(
    keyword: str,
    location: str,
    max_results: int,
    job_type: str,
    work_type: str,
    time_posted: str,
    exp_level: str,
    progress_cb=None,
) -> tuple[list[dict], str]:
    """
    Scrape up to *max_results* jobs from LinkedIn's public search.

    Returns (jobs, error_message).  error_message is "" on full success.
    """
    all_jobs: list[dict] = []
    session = requests.Session()
    session.headers.update(_BROWSER_HEADERS)

    total_pages = (max_results + 24) // 25  # LinkedIn returns 25 per page

    for page in range(total_pages):
        if len(all_jobs) >= max_results:
            break

        url = _build_url(
            keyword, location, page * 25,
            job_type, work_type, time_posted, exp_level,
        )

        if progress_cb:
            progress_cb(page, total_pages, url)

        try:
            resp = session.get(url, timeout=15)
        except requests.exceptions.Timeout:
            return all_jobs, "Request timed out. LinkedIn may be rate-limiting you."
        except requests.exceptions.ConnectionError:
            return all_jobs, "Connection error — check your internet connection."
        except Exception as exc:
            return all_jobs, f"Unexpected error: {exc}"

        # Check for common blocking responses
        if resp.status_code == 429:
            return all_jobs, (
                "LinkedIn rate-limited this request (HTTP 429). "
                "Wait a few minutes and try again."
            )
        if resp.status_code == 403:
            return all_jobs, (
                "LinkedIn blocked the request (HTTP 403). "
                "Try reducing the result count or wait before retrying."
            )
        if resp.status_code != 200:
            return all_jobs, f"LinkedIn returned HTTP {resp.status_code}."

        # Auth-wall / CAPTCHA redirect
        if any(kw in resp.url for kw in ("authwall", "checkpoint", "login")):
            return all_jobs, (
                "LinkedIn redirected to a login wall — the public scraper is "
                "temporarily blocked. Try again in a few minutes."
            )

        soup = BeautifulSoup(resp.text, "lxml")
        page_jobs = _parse_cards(soup)

        if not page_jobs:
            break  # No more results on subsequent pages

        all_jobs.extend(page_jobs)

        # Polite inter-page delay
        if page < total_pages - 1:
            time.sleep(random.uniform(1.5, 3.0))

    return all_jobs[:max_results], ""


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("💼 LinkedIn Job Scraper")
st.caption(
    "Search LinkedIn's public job listings by keyword — no account required."
)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔍 Search")

    keyword = st.text_input(
        "Keywords",
        placeholder="e.g. Python Developer, Data Scientist",
        help="Job title, skill, or any keyword",
    )

    location = st.text_input(
        "Location",
        placeholder="e.g. London, Remote, United States",
        help="City, country, or leave blank for worldwide",
    )

    st.markdown("---")
    st.subheader("Filters")

    max_results = st.slider(
        "Max results",
        min_value=5, max_value=100, value=25, step=5,
        help="LinkedIn serves 25 results per page",
    )
    job_type = st.selectbox("Job type", list(JOB_TYPE_OPTIONS))
    work_type = st.selectbox("Work arrangement", list(WORK_TYPE_OPTIONS))
    exp_level = st.selectbox("Experience level", list(EXP_LEVEL_OPTIONS))
    time_posted = st.selectbox("Date posted", list(TIME_POSTED_OPTIONS))

    st.markdown("---")
    search_btn = st.button("🚀 Search Jobs", type="primary", use_container_width=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "lj_df" not in st.session_state:
    st.session_state.lj_df = None
if "lj_query" not in st.session_state:
    st.session_state.lj_query = ""

# ── Search trigger ────────────────────────────────────────────────────────────

if search_btn:
    if not keyword.strip():
        st.error("Please enter at least one keyword.")
    else:
        prog = st.progress(0, text="Starting…")
        status_ph = st.empty()

        def _on_progress(page: int, total: int, url: str) -> None:
            pct = int(page / total * 100)
            prog.progress(pct, text=f"Fetching page {page + 1} of {total}…")
            status_ph.caption(f"URL: `{url}`")

        jobs, err = scrape_linkedin_jobs(
            keyword=keyword.strip(),
            location=location.strip(),
            max_results=max_results,
            job_type=JOB_TYPE_OPTIONS[job_type],
            work_type=WORK_TYPE_OPTIONS[work_type],
            time_posted=TIME_POSTED_OPTIONS[time_posted],
            exp_level=EXP_LEVEL_OPTIONS[exp_level],
            progress_cb=_on_progress,
        )

        prog.empty()
        status_ph.empty()

        if err:
            st.warning(f"⚠️ {err}")

        if jobs:
            st.session_state.lj_df = pd.DataFrame(jobs)
            loc_label = location.strip() or "Anywhere"
            st.session_state.lj_query = f'"{keyword.strip()}" in {loc_label}'
            st.success(
                f"Found **{len(jobs)}** listings for {st.session_state.lj_query}"
            )
        elif not err:
            st.info(
                "No results found. Try broader keywords, a different location, "
                "or fewer filters."
            )
            st.session_state.lj_df = None

# ── Results display ───────────────────────────────────────────────────────────

if st.session_state.lj_df is not None:
    df: pd.DataFrame = st.session_state.lj_df.copy()

    st.markdown(f"### Results for {st.session_state.lj_query}")

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total listings", len(df))
    m2.metric("Unique companies", df["Company"].nunique())
    remote_n = df["Location"].str.contains("Remote", case=False, na=False).sum()
    m3.metric("Remote positions", int(remote_n))

    st.markdown("---")

    # Filter + sort controls
    fc1, fc2 = st.columns([3, 1])
    with fc1:
        filter_text = st.text_input(
            "Filter",
            placeholder="Filter by title, company, or location…",
            label_visibility="collapsed",
        )
    with fc2:
        sort_col = st.selectbox(
            "Sort by",
            ["Date Posted", "Company", "Title", "Location"],
            label_visibility="collapsed",
        )

    if filter_text:
        mask = df.apply(
            lambda row: filter_text.lower() in row.to_string().lower(), axis=1
        )
        df = df[mask]

    df = df.sort_values(sort_col, ascending=True).reset_index(drop=True)

    view = st.radio("View as", ["Cards", "Table"], horizontal=True)
    st.markdown("---")

    if view == "Cards":
        for _, row in df.iterrows():
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"### {row['Title']}")
                    st.markdown(f"**{row['Company']}** &nbsp;|&nbsp; 📍 {row['Location']}")
                with right:
                    st.caption(f"📅 {row['Date Posted']}")
                    if row["Apply Link"]:
                        st.link_button("View on LinkedIn →", row["Apply Link"])
    else:
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Apply Link": st.column_config.LinkColumn("Apply Link"),
            },
            hide_index=True,
        )

    # CSV download
    st.markdown("---")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download results as CSV",
        data=csv_bytes,
        file_name=f"linkedin_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    # Landing / empty state
    st.info(
        "Use the sidebar to enter a keyword and click **Search Jobs** to get started."
    )
    st.markdown(
        """
        ### How it works

        1. **Enter keywords** — job title, skills, or anything you'd type into LinkedIn's own search bar
           *(e.g. `Machine Learning Engineer`, `React Developer`, `Product Manager`)*
        2. **Set a location** — city, country, or leave blank for worldwide results
        3. **Apply filters** — narrow by job type, work arrangement, experience level, and recency
        4. **Click Search** — the scraper fetches LinkedIn's public, un-authenticated job listings
        5. **Browse & export** — view results as cards or a table, then download as CSV

        ---

        > **Note:** This scraper reads the same public data you'd see without logging in.
        > LinkedIn may occasionally rate-limit requests; if that happens, wait a moment and retry.
        """
    )
