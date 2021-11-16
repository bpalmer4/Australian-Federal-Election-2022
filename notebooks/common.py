# common.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import requests
import bs4

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
    
    
