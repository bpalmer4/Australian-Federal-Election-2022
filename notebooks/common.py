# common.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import requests
import bs4
import re
import statsmodels.api as sm
import stan # install with pip - conda is problematic on the M1 MBP
import os
import time
import datetime

from typing import List


# --- WARNINGS ---

_warnings = []

def warn(message):
    print(message)
    _warnings.append(message)


def print_warnings():
    for i, w in enumerate(_warnings):
        print(f'{i+1:3d}: {w}')


def check_file_current(filename, message):
    file_status = os.stat(filename)
    modified_date = datetime.date(*time.localtime(file_status.st_mtime)[:3])
    today = datetime.date.today()
    if modified_date != today:
        warn(f'{filename}: File looks old. ' + message)

        
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


# --- POLL AGGREGATION ---

# Calulcate a LOWESS regression
def get_lowess(votes, dates, period=60):

    day = (dates - dates.min()) / pd.Timedelta(days=1) + 1
    frac = period / day.max()
    lowess = sm.nonparametric.lowess(
        endog=votes, exog=day, # y, x ...
        frac=frac, is_sorted=True)

    lowess = {int(x[0]): x[1] for x in lowess}
    lowess = day.map(lowess).interpolate()
    lowess.index = dates

    return lowess, period


# Bayesian aggregation of vote time-series data
def bayes_poll_aggregation(df, 
                           poll_column=None,
                           date_column=None,
                           firm_column=None,
                           assumed_sample_size=1_000,
                           num_chains=4,
                           num_samples=2_500):
    """Calculate a Bayesian aggregation for a series of polling results"""
    
    # initial sanity checks
    assert df[poll_column].notna().all()
    assert df[date_column].notna().all()
    assert df[firm_column].notna().all()
    assert (df[poll_column] >= 0).all()
    assert (df[poll_column] <= 100).all()
    
    # preparation
    print(f'Stan version: {stan.__version__}')
    df = df.copy() # do no harm
    
    pseudoSampleSigma = np.sqrt((50 * 50) / assumed_sample_size)
    
    first_day = df['Mean Date'].min()
    df['_Day'] = (
        ((df[date_column] - first_day) 
         / pd.Timedelta(days=1)).astype(int) 
        + 1
    )
    
    df[firm_column] = df[firm_column].astype('category')
    df['_House'] = df[firm_column].cat.codes + 1
    brand_map = {x+1: y for x, y in zip(df[firm_column].cat.codes, df[firm_column])}

    model_data = {
        'n_polls': len(df),
        'n_days': int(df['_Day'].max()),
        'n_houses': int(df['_House'].max()),
        'centre_offset': df[poll_column].mean(),

        'pseudoSampleSigma': pseudoSampleSigma,
    
        'y': df[poll_column].to_list(),
        'house': df['_House'].to_list(),
        'day': df['_Day'].to_list(),
    }
    # print(model_data)
    
    # load model
    with open('2pp.stan') as f:
        model_code = f.read()
        
    # run model
    posterior = stan.build(model_code, data=model_data)
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples)
    
    return fit, first_day, brand_map


def bayes_poll_aggregation_plots(df, 
                                 fit, 
                                 first_day, 
                                 brand_map,
                              poll_column,
                              date_column,
                              firm_column,
                                 party,
                                 title,
                                 line_color,
                                 point_color,
                                s_args):
    
    # This is a bit of a hack, and certainly too long a function
    
    # a framework for quantifying where the samples lie
    quants = [0.005, 0.025, 0.100, 0.250, 0.500, 0.750, 0.900, 0.975, 0.995]
    LOW, HIGH = 'low', 'high'
    ranges = pd.DataFrame({
        '99%': (0.005, 0.995),
        '95%': (0.025, 0.975),
        '80%': (0.100, 0.900),
        '50%': (0.250, 0.750),
    }, index=[LOW, HIGH]).T

    # results in DataFrame
    results_df = fit.to_frame()
    
    # Get the daily hidden vote share data
    hvs = (
        results_df[
            results_df.columns[
                results_df.columns.str.contains('hidden_vote_share')
            ]
        ]
    )
    hvs.columns = pd.date_range(start=first_day, freq='D', 
                                periods=len(hvs.columns))
    hvs = hvs.quantile(quants)
    
    # plot this daily hidden vote share
    fig, ax = initiate_plot()
    alpha = 0.1
    for x, y in ranges.iterrows():
        low = y[0]
        high = y[1]
        lowpoint = hvs.loc[low]
        highpoint = hvs.loc[high]
        ax.fill_between(x=lowpoint.index, y1=lowpoint, y2=highpoint,
                        color=line_color, alpha = alpha,label=x,)
        alpha += 0.075
    
    ax.plot(hvs.columns, hvs.loc[0.500], 
            color=line_color, lw=1, label='Median')

    # annotate end-point median to one devimal place ...
    ax.text(hvs.columns[-1] + pd.Timedelta(days=10), 
            hvs.loc[0.500][-1], 
            f'{hvs.loc[0.500].round(1)[-1]}',
            rotation=90, ha='left', va='center',
            fontsize=14)

    BENCHMARK = 50
    span = ax.get_ylim()
    if span[0] <= BENCHMARK <= span[1]:
        ax.axhline(y=BENCHMARK, c='#999999', lw=0.5)

    markers = ['x', '+', '1', '2', '3', '4', '<', '>', '^', 'v', 'o', 's', '*', ]
    for i, brand in enumerate(sorted(df[firm_column].unique())):
        subset = df[df[firm_column] == brand].copy()
        a = subset[date_column]
        b = subset[poll_column]
        #print('DEBUG', len(subset), len(a), len(b), type(a), type(b), a, b)
        #display(subset)
        if not len(subset):
            continue # ignore empty subsets
        ax.scatter(a, b, marker=markers[i], label=brand, color=point_color)

    ax.legend(loc='best', ncol=2)
    
    plot_finalise(ax, 
                  title=f'{party} {title}',
                  ylabel=f'Per cent vote share for {party}',
                  **s_args,
                 )

    # get the house effects data
    house_effects = results_df[results_df.columns[results_df.columns.str.contains('houseEffect')]]

    # map the ugly column names back to something meaningful
    house_effects.columns = (
        house_effects.columns
        .str.extract(r'([\d]+)$')
        .pipe(lambda x: x[x.columns[0]])
        .astype(int)
        .map(brand_map)
    )

    # get sample quants
    house_effects = house_effects.quantile(quants)
    
    # plot the house effects data
    fig, ax = initiate_plot()

    for i, house in enumerate(house_effects.columns):
        alpha = 0.1
        for x, y in ranges.iterrows():
            low = y[0]
            high = y[1]
            lowpoint = house_effects.loc[low, house]
            width = house_effects.loc[high, house] - lowpoint
            label = x if i == 0 else None
            ax.barh(y=house, left=lowpoint, width=width, 
                    height=0.5, color=line_color, alpha=alpha,
                    label=label)
            alpha += 0.075

    ax.scatter(y=range(len(house_effects.columns)), 
               x=house_effects.loc[0.500],
               marker='d', facecolor='white',
               edgecolor=line_color, 
               linewidth=0.5, zorder=2,
               label='Median', s=90)

    print(house_effects.loc[0.500])
    ax.legend(loc='best')

    plot_finalise(ax, 
                         title=f'{party} {title} (House Effects)',
                         xlabel=f'Percentage Points (towards {party})',
                         **s_args, )
    
    
# --- PLOTTING ---

COLOR_COALITION = 'darkblue'
COLOR_LABOR = 'darkred'
COLOR_OTHER = 'darkorange'
COLOR_GREEN = 'darkgreen'

P_COLOR_COALITION = '#0000dd'
P_COLOR_LABOR = '#dd0000'
P_COLOR_OTHER = 'orange'
P_COLOR_GREEN = 'green'


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
    
    
