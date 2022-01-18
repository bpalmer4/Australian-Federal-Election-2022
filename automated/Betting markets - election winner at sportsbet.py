#!/usr/bin/env python
# coding: utf-8

# # Betting markets - election winner at sportsbet
# 
# Note: this notebook is for ease of testing. Convert to a python file and move to the automated directory.
# 
# To do this ...
# ```
# ipython nbconvert --to python "Betting markets - election winner at sportsbet.ipynb"
# chmod 700 "Betting markets - election winner at sportsbet.py"
# mv "Betting markets - election winner at sportsbet.py" ../automated ```

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Python-setup" data-toc-modified-id="Python-setup-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Python setup</a></span></li><li><span><a href="#Set-up-web-driver-options" data-toc-modified-id="Set-up-web-driver-options-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Set-up web-driver options</a></span></li><li><span><a href="#Extract-website-text-using-Selenium" data-toc-modified-id="Extract-website-text-using-Selenium-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Extract website text using Selenium</a></span></li><li><span><a href="#Extract-data-of-interest" data-toc-modified-id="Extract-data-of-interest-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Extract data of interest</a></span></li><li><span><a href="#Append-this-data-to-a-CSV-file" data-toc-modified-id="Append-this-data-to-a-CSV-file-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Append this data to a CSV file</a></span></li><li><span><a href="#Final-sanity-check" data-toc-modified-id="Final-sanity-check-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Final sanity check</a></span></li></ul></div>

# ## Python setup

# In[1]:


# data science imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# web scraping imports
from selenium import webdriver
from bs4 import BeautifulSoup

# CSV utilities
from csv import DictWriter

# System imports
import re
import datetime
from pathlib import Path


# ## Set-up web-driver options

# In[2]:


options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')


# ## Extract website text using Selenium

# In[3]:


# get the web page text
driver = webdriver.Chrome(options=options)
driver.implicitly_wait(220) 
url = (
    'https://www.sportsbet.com.au/betting/politics/'
    'australian-federal-politics/Next-Federal-Election-Type-of-Government-Formed-5758351'
)
driver.get(url)
soup = BeautifulSoup(driver.page_source, 'lxml')
driver.close()
#print(soup.prettify())


# ## Extract data of interest

# In[4]:


# extract the data of interest
div_name = "content-background" # this looks fragile
div = soup.find("div", {"data-automation-id": div_name})
class_name  = "outcomeDetails_f1t3f12" # this looks fragile
odds = div.find_all("div", {"class": class_name})
pattern = r"([^\d]+)([\d\.]+)"
comp_pattern = re.compile(pattern)
found = {}
for c in odds:
    match = re.search(comp_pattern, c.text)
    found[match[1]] = match[2]
#found['Date'] = datetime.datetime.now()


# In[5]:


# long format assert False


# ## Append this data to a CSV file

# In[6]:


# long format
df = pd.DataFrame([found.keys(), found.values()], index=['variable', 'value']).T
df.index =np.repeat(datetime.datetime.now(), len(df))
df.index.name = 'datetime'


# In[7]:


# save to file
FILE = '../historical-data/sportsbet-2022-outcome.csv'
df.to_csv(FILE, mode='a', index=True, header=False)


# In[ ]:




