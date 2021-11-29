# common.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import requests
import bs4
import re

from typing import List


# --- WEB BASED DATA CAPTURE --

def get_url_text(url: str):
    response = requests.get(url)
    assert(response.status_code == 200) # successful retrieval
    return response.text


def get_tables(text):
    soup = bs4.BeautifulSoup(text, features="lxml")
    tables = soup.findAll('table')
    return tables


def get_table_from_text(number, text):
    tables = get_tables(text)
    html = str(tables[number])
    df = pd.read_html(html, flavor='bs4', na_values='–')[0]
    df = df.dropna(axis=0, how='all') # remove empty rows
    df = df.dropna(axis=1, how='all') # remove empty columns
    return df


# --- DATA CLEANING ---

# Common unicode symbols
endash = '\u2013'
emdash = '\u2014'
hyphen = '\u002D'
minus  = '\u2212'
tilde =  '~'
comma =  ','


def remove_event_rows(t: pd.DataFrame) -> pd.DataFrame:
    """Remove the event marker rows."""
    t = t.loc[t[t.columns[0]] != t[t.columns[1]]]
    t = t.loc[t[t.columns[1]] != t[t.columns[2]]]
    t = t[t[t.columns[1]].notna()]
    return t


def drop_empty(t: pd.DataFrame) -> pd.DataFrame:
    """Remove all empty rows and columns."""
    t = t.dropna(axis=0, how='all')
    t = t.dropna(axis=1, how='all')
    return t


def fix_numerical_cols(t: pd.DataFrame) -> pd.DataFrame:
    """Convert selected columns from strings to numeric data type."""
    
    # some constants
    fixable_cols = (
        # For the 2022 cycle
        'Primary vote', '2pp vote',
        'Preferred Prime Minister',
        'Morrison', 'Albanese',
        'Sample size', 
        # For historical cycles
        'TPP vote', '2PP vote', 
        'Political parties',
        'Two-party-preferred',
    )

    # navigate fixable columns and convert to numerical dtype
    for c in t.columns[t.columns.get_level_values(0).isin(fixable_cols)]:
        if str(t[c].dtype) in ['object', ]:
            t[c] = (
                t[c]
                .str.replace('\[.*\]$', '', regex=True) # remove footnotes
                .str.replace('%', '')    # remove percent symbol
                .str.replace(tilde, '')  # replace tilde with nothing
                .str.replace(hyphen, '') # replace hyphen with nothing
                .str.replace(endash, '') # replace hyphen with nothing
                .str.replace(emdash, '') # replace hyphen with nothing
                .str.replace(minus, '')  # replace hyphen with nothing
                .str.replace('n/a', '')  # replace 'n/a' with nothing
                .str.replace('?', '', regex=False)  # replace '?' with nothing
                .str.replace('<', '', regex=False)  # replace '<' with nothing
                .str.strip()             # strip white space
                .replace('', np.nan)     # NaN empty lines
                .astype(float)           # float
            )
    return t


def fix_column_names(t: pd.DataFrame) -> pd.DataFrame:
    """Replace 'Unnamed' column names with ''."""
    
    replacements = {}
    for c in t.columns:
        if 'Unnamed' in c[1]:
            replacements[c[1]] = ''
    if replacements:
        t = t.rename(columns=replacements, level=1)
    return t


def remove_footnotes(t: pd.DataFrame) -> pd.DataFrame:
    """Remove Wikipedia footnote references from the Brand column"""
    
    BRAND = ['Brand', 'Firm']
    
    for brand in BRAND:
        if brand not in t.columns.get_level_values(0):
            continue
        col = t.columns[t.columns.get_level_values(0) == brand]
        assert(len(col) == 1)
        t.loc[:, col] = (
            t.loc[:, col] # returns a single column DataFrame
            .pipe(lambda x: x[x.columns[0]]) # make as Series
            .str.replace('\[.*\]$', '', regex=True) # remove footnotes
            .str.strip() # remove any leading/trailing whitespaces
        )
    return t


def get_mean_date(tokens: List[str]) -> pd.Timestamp:
    """Extract the middle date from a list of date tokens."""
    
    last_day = None
    day, month, year = None, None, None
    remember = tokens.copy()
    while tokens:
        token = tokens.pop()
        if re.match(r'[0-9]{4}', token):
            year = token
        elif re.match(r'[A-Za-z]+', token):
            month = token
        elif re.match(r'[0-9]{1,2}', token):
            day = token
        else:
            print(f'Warning: {token} not recognised in get_mean_date()'
                  f'with these date tokens {remember}')    
            
        if (last_day is None and day is not None 
            and month is not None and year is not None):
            last_day = pd.Timestamp(f'{year} {month[:3]} {day}')
    
    #
    if month is None:
        print(f'Warning: missing month in these tokens? {remember}')
    
    # sadly we have cases of this ...
    if not last_day:
        if day is None:
            day = 15 # middle of month
        last_day = pd.Timestamp(f'{year} {month[:3]} {day}')

    # get the middle date
    first_day = pd.Timestamp(f'{year} {month[:3]} {day}')
    if first_day > last_day:
        print(f'Check these dates in get_mean_date(): {first_day} '
              f'{last_day} with these tokens {remember}')
    
    return (first_day + ((last_day - first_day) / 2)).date()


def tokenise_dates(dates: pd.Series) -> pd.Series:
    """Return the date as a list of tokens."""
    return (
        dates
        .str.replace(endash, hyphen)
        .str.replace(emdash, hyphen)
        .str.replace(minus, hyphen)
        .str.replace('c. ', '', regex=False)
        .str.split(r'[\-,\s\/]+')
    )


def middle_date(t: pd.DataFrame) -> pd.DataFrame:
    """Get the middle date in the date range, into column 'Mean Date'."""
    
    # assumes dates in strings are ordered from first to last
    tokens = tokenise_dates(t['Date'])
    t['Mean Date'] = tokens.apply(get_mean_date).astype('datetime64[ns]')
    return t


def clean(table: pd.DataFrame) -> pd.DataFrame:
    """Clean the extracted data tables."""
    
    t = table.copy()
    t = remove_event_rows(t)
    t = drop_empty(t)
    t = fix_numerical_cols(t)
    t = fix_column_names(t)
    t = remove_footnotes(t)
    t = middle_date(t)
    t = t.set_index(('Mean Date', ''))
    t = t.sort_index(ascending=True)
    # Note we keep the hierarchical index at this point
    # because it makes the checking row additions simpler
    
    return t


def flatten_col_names(columns: pd.Index) -> List[str]:
    """Flatten the hierarchical column index."""
    
    return [' '.join(col).strip() for col in columns.values]


# --- PLOTTING ---

COLOR_COALITION = 'darkblue'
COLOR_LABOR = '#dd0000'
COLOR_OTHER = 'darkorange'
COLOR_GREEN = 'darkgreen'


def initiate_plot():
    """Get a matplotlib figure and axes instance."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.margins(0.02)
    return fig, ax


def plot_finalise(ax, title=None, xlabel=None, ylabel=None, 
                  lfooter=None, rfooter='marktheballot.blogspot.com',
                  location='../charts/'):
    """Complete and save a plot image"""
    
    # annotate the plot
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if lfooter is not None:
        ax.figure.text(0.005, 0.005, lfooter, 
                       ha='left', va='bottom',
                       c='#999999', style='italic', 
                       fontsize=8)

    if rfooter is not None:
        ax.figure.text(0.995, 0.005, rfooter, 
                       ha='right', va='bottom',
                       c='#999999', style='italic', 
                       fontsize=8)
        
    ax.figure.tight_layout(pad=1.1)    
    
    if title is not None:
        pathlib.Path(location).mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(location+title+'.png', dpi=300)
    
    # close
    plt.show()
    plt.close()
    
    
