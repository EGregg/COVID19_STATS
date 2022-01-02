#Project JEMA
#Jonathan Fillipini, Edward Gregg, Michelle Nguyen, Andrew Christian
#CMSC 206-
#Final Group Project
#12/8/2021

##################################################
#install required libraries
!pip install datetime
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install requests
!pip install urllib3
#!pip install json
#!pip install csv
!pip install wordcloud
!pip install plotly
#!pip install re
!pip install bs4
!pip install pillow
!pip install seaborn
!pip install folium
#!pip install math

##################################################
#import libraries
from datetime import date, timedelta
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
import requests
import urllib.request as ul, urllib.parse, urllib.error
import json
from pandas import json_normalize
import csv
from wordcloud import WordCloud, STOPWORDS
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from plotly.graph_objs import Line
import re
import bs4 as bs
from PIL import Image
import plotly.graph_objects as go
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.figure_factory as ff
from plotly.colors import n_colors
import folium
from folium import Marker
from folium.plugins import MarkerCluster
import math

####################################################################################################
## functions #######################################################################################
####################################################################################################
#NOTE: each question is treated as a single function

####################################################################################################
def menu(): ########################################################################################
####################################################################################################

  print('\n###################')
  print('## Question MENU ##')
  print('###################\n')
  print('1. How does income affect vaccination rates amongst countries?')
  print('2. What are the implications of the vaccination rates amongst countries?')
  print('3. Does the homeless population have any correlation with vaccination rates?')
  print('4. Do states with the highest COVID rate influence the states around them?')
  print('5. Could the rate of COVID be slowed in a highly controlled environment (state prison) vs a less controlled environment (states)?')
  print('6. How much of an issue was COVID-19 in the last presedential debate series?')
  print('7. As vaccinations continue to be administered to the general public, have the numbers of infected unvaccinated people gone down or up?')
  print('8. What are the demographics of the people who visit the places where infection outbreaks occur?')
  print('9. Are there any racial disparities in DC when it comes to COVID data?')
  print('10. What relationhisp is there between Covid-19 infections and unemployment claims?')
  print('11. What relationhsip is there between Covid-19 vaccination rates and unemployment claims?')
  print('12. How significant of an impact did Covid-19 have on unempoyment claims?')
  print('13. EXIT')
  print()

####################################################################################################
def question1(): ###################################################################################
####################################################################################################
#How does income affect vaccination rates amongst countries?

  print()
  print('################')
  print("## Question 1 ##")
  print('################\n')
  print('Q. How does income affect vaccination rates amongst countries?\n')

  #sns.set(color_codes = True)
  #sns.set(style="whitegrid")

  df = pd.read_csv('country_vaccinations.csv')
  display(df.head())

  df.fillna(value = 0, inplace = True) #Fill in all the null value
  df.total_vaccinations = df.total_vaccinations.astype(int)
  df.people_vaccinated = df.people_vaccinated.astype(int)
  df.people_fully_vaccinated = df.people_fully_vaccinated.astype(int)
  df.daily_vaccinations_raw = df.daily_vaccinations_raw.astype(int)
  df.daily_vaccinations = df.daily_vaccinations.astype(int)
  df.total_vaccinations_per_hundred = df.total_vaccinations_per_hundred.astype(int)
  df.people_fully_vaccinated_per_hundred = df.people_fully_vaccinated_per_hundred.astype(int)
  df.daily_vaccinations_per_million = df.daily_vaccinations_per_million.astype(int)
  df.people_vaccinated_per_hundred = df.people_vaccinated_per_hundred.astype(int)
  date = df.date.str.split('-', expand =True)
  display(date) #Here we are cleaning up the code

  df['year'] = date[0]
  df['month'] = date[1]
  df['day'] = date[2]
  df.year = pd.to_numeric(df.year)
  df.month = pd.to_numeric(df.month)
  df.day = pd.to_numeric(df.day)
  df.date = pd.to_datetime(df.date)
  display(df.head())

  display(df.country.unique())
  #unique() function in pandas helps to get unique values present in the feature.

  def size(m,n):
    fig = plt.gcf();
    fig.set_size_inches(m,n);

  wordCloud = WordCloud(
    background_color='white',
    max_font_size = 50).generate(' '.join(df.country))
  plt.figure(figsize=(15,7))
  plt.axis('off')
  plt.imshow(wordCloud)
  plt.show()

  country_wise_total_vaccinated = {}
  for country in df.country.unique() :
    vaccinated = 0
    for i in range(len(df)) :
      if df.country[i] == country :
        vaccinated += df.daily_vaccinations[i]
    country_wise_total_vaccinated[country] = vaccinated
  # made a seperate dict from the df
    country_wise_total_vaccinated_df = pd.DataFrame.from_dict(country_wise_total_vaccinated,
                                                         orient='index',
                                                         columns = ['total_vaccinted_till_date'])
  # converted dict to df
  country_wise_total_vaccinated_df.sort_values(by = 'total_vaccinted_till_date', ascending = False, inplace = True)
  display(country_wise_total_vaccinated_df)

  fig = px.bar(country_wise_total_vaccinated_df,
             y = 'total_vaccinted_till_date',
             x = country_wise_total_vaccinated_df.index,
             color = 'total_vaccinted_till_date',
             color_discrete_sequence= px.colors.sequential.Viridis_r
            )
  fig.update_layout(
    title={
            'text' : "Vaccination till date in various countries",
            'y':0.95,
            'x':0.5
        },
    xaxis_title="Countries",
    yaxis_title="Total vaccinated",
  )
  fig.show()

  fig = px.line(df, x = 'date', y ='daily_vaccinations', color = 'country')
  fig.update_layout(
    title={
            'text' : "Daily vaccination trend",
            'y':0.95,
            'x':0.5
        },
    xaxis_title="Date",
    yaxis_title="Daily Vaccinations"
  )
  fig.show()

####################################################################################################
def question2(): ###################################################################################
####################################################################################################
#What are the implications of the vaccination rates amongst countries?

  print()
  print('################')
  print("## Question 2 ##")
  print('################\n')
  print('Q. What are the implications of the vaccination rates amongst countries?\n')

  #sns.set(color_codes = True)
  #sns.set(style="whitegrid")

  df = pd.read_csv('country_vaccinations.csv')
  #display(df.head())

  df.fillna(value = 0, inplace = True) #Fill in all the null value
  df.total_vaccinations = df.total_vaccinations.astype(int)
  df.people_vaccinated = df.people_vaccinated.astype(int)
  df.people_fully_vaccinated = df.people_fully_vaccinated.astype(int)
  df.daily_vaccinations_raw = df.daily_vaccinations_raw.astype(int)
  df.daily_vaccinations = df.daily_vaccinations.astype(int)
  df.total_vaccinations_per_hundred = df.total_vaccinations_per_hundred.astype(int)
  df.people_fully_vaccinated_per_hundred = df.people_fully_vaccinated_per_hundred.astype(int)
  df.daily_vaccinations_per_million = df.daily_vaccinations_per_million.astype(int)
  df.people_vaccinated_per_hundred = df.people_vaccinated_per_hundred.astype(int)
  date = df.date.str.split('-', expand =True)
  #display(date) #Here we are cleaning up the code

  df['year'] = date[0]
  df['month'] = date[1]
  df['day'] = date[2]
  df.year = pd.to_numeric(df.year)
  df.month = pd.to_numeric(df.month)
  df.day = pd.to_numeric(df.day)
  df.date = pd.to_datetime(df.date)

  india_usa = [df[df.country == 'United States'], df[df.country == 'India']]
  result = pd.concat(india_usa)
  fig = px.line(result, x = 'date', y ='total_vaccinations', color = 'country')
  fig.update_layout(
    title={
            'text' : "Total vaccinated - India vs USA",
            'y':0.95,
            'x':0.5
        },
    xaxis_title="Date",
    yaxis_title="Total Vaccinations"
  )
  fig.show()

####################################################################################################
def question3(): ###################################################################################
####################################################################################################
#Does the homeless population have any correlation with vaccination rates?

  print()
  print('################')
  print("## Question 3 ##")
  print('################\n')
  print('Q. Does the homeless population have any correlation with vaccination rates?\n')

  #sns.set(color_codes = True)
  #sns.set(style="whitegrid")

  # Population Data
  populationData = pd.read_csv('2019_Census_US_Population_Data_By_State_Lat_Long.csv')
  freshDate = date.today() - timedelta(days=1)
  freshDate = date.strftime(freshDate,"%Y%m%d")
  freshDate = freshDate[0:4] + "-" + freshDate[4:6] + "-" + freshDate[6:8]
  vaccinationData = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/us_state_vaccinations.csv')
  vaccinationByLocation = vaccinationData.loc[(vaccinationData.date == freshDate)][["location", "people_vaccinated"]]

  # Vaccination and population data
  vaccinationAndPopulationByLocation = pd.merge(populationData, vaccinationByLocation, left_on='STATE',right_on='location').drop(columns="location")

  # Calculate percentage vaccinated by state
  vaccinationAndPopulationByLocation["percent_vaccinated"] = vaccinationAndPopulationByLocation["people_vaccinated"] / vaccinationAndPopulationByLocation["POPESTIMATE2019"]

  display(vaccinationAndPopulationByLocation)

  homeless = pd.read_csv("2007-2016-Homelessnewss-USA.csv")
  population = pd.read_csv("Population-by-state.csv")

  pop = population.copy()
  pop.columns = pop.iloc[0]
  pop.drop(0,axis=0, inplace=True)
  pop.drop(['Id', 'Id2','April 1, 2010 - Census', 'April 1, 2010 - Estimates Base'], axis=1, inplace=True)
  pop.columns = ['State','pop10','pop11','pop12','pop13','pop14','pop15','pop16']

  """Clean homelss dataset"""
  hless = homeless.copy()
  hless['Count'] = hless['Count'].str.replace(',', '').astype(np.int64) #turn count number into int
  hless.drop(['CoC Number','CoC Name'], axis=1, inplace=True)
  hless['Year'] = pd.to_datetime(hless['Year'])
  hless['Year'] = hless['Year'].dt.year
  display(hless.head())

  st_hless = hless.copy()
  st_pop = pop.copy()
  st_pop['State'] = st_pop['State'].replace({'Alaska':'AK', 'Alabama':'AL', 'Arkansas':'AR', 'Arizona':'AZ',
                                                                 'California':'CA', 'Colorado':'CO', 'Connecticut':'CT',
                      'District of Columbia':'DC', 'Delaware':'DE', 'Florida':'FL',
                      'Georgia':'GA', 'Hawaii':'HI', 'Iowa':'IA',
                      'Idaho':'ID', 'Illinois':'IL', 'Indiana':'IN', 'Kansas':'KS',
                      'Kentucky':'KY', 'Louisiana':'LA', 'Massachusetts':'MA', 'Maryland':'MD',
                      'Maine':'ME', 'Michigan':'MI', 'Minnesota':'MN', 'Missouri':'MO',
                      'Mississippi':'MS', 'Montana':'MT', 'North Carolina':'NC',
                      'North Dakota':'ND', 'Nebraska':'NE', 'New Hampshire':'NH',
                      'New Jersey':'NJ', 'New Mexico':'NM', 'Nevada':'NV', 'New York':'NY',
                      'Ohio':'OH', 'Oklahoma':'OK', 'Oregon':'OR', 'Pennsylvania':'PA',
                      'Puerto Rico':'PR', 'Rhode Island':'RI', 'South Carolina':'SC',
                      'South Dakota':'SD', 'Tennessee':'TN', 'Texas':'TX', 'Utah':'UT',
                      'Virginia':'VA', 'Vermont':'VT', 'Washington':'WA',
                      'Wisconsin':'WI', 'West Virginia':'WV', 'Wyoming':'WY'})
  st_hless = st_hless[(st_hless['State']!= 'GU') & (st_hless['State']!= 'VI')]
  st_hless =  st_hless[(st_hless['Year']!= 2007) & (st_hless['Year']!= 2008) &
                      (st_hless['Year']!= 2009)]
  homelessness = st_hless.merge(st_pop, on='State')
  display(homelessness.head())

  hlessness = homelessness.copy()
  hlessness = hlessness[hlessness['Measures']=='Total Homeless']
  hlessness = hlessness[(hlessness['Year']==2010) | (hlessness['Year']==2016)]
  hlessness = hlessness.groupby(['State','Year',])[['Count']].sum()
  hlessness.reset_index(inplace=True)

  hlessness = hlessness.merge(st_pop, on='State')
  hlessness['pop10'] = hlessness['pop10'].astype(int)
  hlessness['pop16'] = hlessness['pop16'].astype(int)

  p10 = hlessness.where(hlessness['Year']==2010)
  p16 = hlessness.where(hlessness['Year']==2016)
  p10['pop_percnt10'] = p10['Count']/p10['pop10']
  p16['pop_percnt16'] = p16['Count']/p16['pop16']
  hlessness = p10.combine_first(p16.rename(columns={'pop_percnt16':'pop_percnt10'}))
  hlessness = hlessness.rename(columns={'pop_percnt10':'%_of_pop'})

  percent = hlessness[hlessness['Year']==2016]
  print ("1.2 out of every 100 people in DC is homeless\n\n\
  Missourri has lowest % of homeless, with less than 1 out of every 1000\n\n\
  In New York, almost 1 out of every 200 is homeless")
  plt.figure(figsize=(18,6))
  sns.barplot(y='%_of_pop', x='State', data=percent)
  plt.ylabel('Percentage')
  plt.title('Percentage of homeless from state population')

  plt.show()

####################################################################################################
def question4(): ###################################################################################
####################################################################################################
#Do states with the highest COVID rate influence the states around them?

  print()
  print('################')
  print("## Question 4 ##")
  print('################\n')
  print('Q. Do states with the highest COVID rate influence the states around them?\n')

  #read the data and create a dataframe
  file = 'us-states.csv'
  df = pd.read_csv(file)

  #This groups each state together and then finds the max of the state in each column
  grouped = df.groupby('state', sort=True).max().reset_index()
  print(grouped)

  #Shows a graph with the total amount of cases per state
  grouped.plot.bar(x="state", y="cases", title="Number of total COVID-19 cases");
  plt.show(block=True);


  grp = df.groupby('state')

  bordering_alabama = grp.get_group('Florida').max()['cases'] + grp.get_group('Georgia').max()['cases'] + grp.get_group('Mississippi').max()['cases'] + grp.get_group('Tennessee').max()['cases']
  bordering_alaska = grp.get_group('Alaska').max()['cases']
  bordering_arizona = grp.get_group('California').max()['cases'] + grp.get_group('Colorado').max()['cases'] + grp.get_group('Nevada').max()['cases'] + grp.get_group('New Mexico').max()['cases'] + grp.get_group('Utah').max()['cases']
  bordering_Arkansas = grp.get_group('Louisiana').max()['cases'] + grp.get_group('Mississippi').max()['cases'] + grp.get_group('Missouri').max()['cases'] + grp.get_group('Oklahoma').max()['cases'] + grp.get_group('Tennessee').max()['cases'] + grp.get_group('Texas').max()['cases']
  bordering_California = grp.get_group('Arizona').max()['cases'] + grp.get_group('Nevada').max()['cases'] + grp.get_group('Oregon').max()['cases']
  bordering_Colorado = grp.get_group('Arizona').max()['cases'] + grp.get_group('Kansas').max()['cases'] + grp.get_group('Nebraska').max()['cases'] + grp.get_group('New Mexico').max()['cases'] + grp.get_group('Oklahoma').max()['cases'] + grp.get_group('Utah').max()['cases'] + grp.get_group('Wyoming').max()['cases']
  bordering_Connecticut = grp.get_group('Massachusetts').max()['cases'] + grp.get_group('New York').max()['cases'] + grp.get_group('Rhode Island').max()['cases']
  bordering_Delaware = grp.get_group('New Jersey').max()['cases'] + grp.get_group('Pennsylvania').max()['cases'] + grp.get_group('Maryland').max()['cases']
  bordering_florida = grp.get_group('Florida').max()['cases'] + grp.get_group('Alabama').max()['cases'] + grp.get_group('Georgia').max()['cases']
  bordering_Georgia = grp.get_group('North Carolina').max()['cases'] + grp.get_group('South Carolina').max()['cases'] + grp.get_group('Tennessee').max()['cases'] + grp.get_group('Alabama').max()['cases'] + grp.get_group('Florida').max()['cases']
  bordering_Hawaii = 0
  bordering_Idaho = grp.get_group('Utah').max()['cases'] + grp.get_group('Washington').max()['cases'] + grp.get_group('Wyoming').max()['cases'] + grp.get_group('Montana').max()['cases'] + grp.get_group('Nevada').max()['cases'] + grp.get_group('Oregon').max()['cases']
  bordering_Illinois = grp.get_group('Kentucky').max()['cases'] + grp.get_group('Missouri').max()['cases'] + grp.get_group('Wisconsin').max()['cases'] + grp.get_group('Indiana').max()['cases'] + grp.get_group('Iowa').max()['cases'] + grp.get_group('Michigan').max()['cases']
  bordering_Indiana = grp.get_group('Michigan').max()['cases'] + grp.get_group('Ohio').max()['cases'] + grp.get_group('Illinois').max()['cases'] + grp.get_group('Kentucky').max()['cases']
  bordering_Iowa = grp.get_group('Nebraska').max()['cases'] + grp.get_group('South Dakota').max()['cases'] + grp.get_group('Wisconsin').max()['cases'] + grp.get_group('Illinois').max()['cases'] + grp.get_group('Minnesota').max()['cases'] + grp.get_group('Missouri').max()['cases']
  bordering_Kansas = grp.get_group('Nebraska').max()['cases'] + grp.get_group('Oklahoma').max()['cases'] + grp.get_group('Colorado').max()['cases'] + grp.get_group('Missouri').max()['cases']
  bordering_Kentucky = grp.get_group('Tennessee').max()['cases'] + grp.get_group('Virginia').max()['cases'] + grp.get_group('West Virginia').max()['cases'] + grp.get_group('Illinois').max()['cases'] + grp.get_group('Indiana').max()['cases'] + grp.get_group('Missouri').max()['cases'] + grp.get_group('Ohio').max()['cases']
  bordering_Louisiana = grp.get_group('Texas').max()['cases'] + grp.get_group('Arkansas').max()['cases'] + grp.get_group('Mississippi').max()['cases']
  bordering_Maine = grp.get_group('New Hampshire').max()['cases']
  bordering_Maryland = grp.get_group('Virginia').max()['cases'] + grp.get_group('West Virginia').max()['cases'] + grp.get_group('Delaware').max()['cases'] + grp.get_group('Pennsylvania').max()['cases']
  bordering_Massachusetts = grp.get_group('New York').max()['cases'] + grp.get_group('Rhode Island').max()['cases'] + grp.get_group('Vermont').max()['cases'] + grp.get_group('Connecticut').max()['cases'] + grp.get_group('New Hampshire').max()['cases']
  bordering_Michigan = grp.get_group('Ohio').max()['cases'] + grp.get_group('Wisconsin').max()['cases'] + grp.get_group('Indiana').max()['cases'] + grp.get_group('Minnesota').max()['cases']
  bordering_Minnesota = grp.get_group('North Dakota').max()['cases'] + grp.get_group('South Dakota').max()['cases'] + grp.get_group('Wisconsin').max()['cases'] + grp.get_group('Iowa').max()['cases'] + grp.get_group('Michigan').max()['cases']
  bordering_Mississippi = grp.get_group('Louisiana').max()['cases'] + grp.get_group('Tennessee').max()['cases'] + grp.get_group('Alabama').max()['cases'] + grp.get_group('Arkansas').max()['cases']
  bordering_Missouri = grp.get_group('Nebraska').max()['cases'] + grp.get_group('Oklahoma').max()['cases'] + grp.get_group('Tennessee').max()['cases'] + grp.get_group('Arkansas').max()['cases'] + grp.get_group('Illinois').max()['cases'] + grp.get_group('Iowa').max()['cases'] + grp.get_group('Kansas').max()['cases'] + grp.get_group('Kentucky').max()['cases']
  bordering_Montana = grp.get_group('South Dakota').max()['cases'] + grp.get_group('Wyoming').max()['cases'] + grp.get_group('Idaho').max()['cases'] + grp.get_group('North Dakota').max()['cases']
  bordering_Nebraska = grp.get_group('Missouri').max()['cases'] + grp.get_group('South Dakota').max()['cases'] + grp.get_group('Wyoming').max()['cases'] + grp.get_group('Colorado').max()['cases'] + grp.get_group('Iowa').max()['cases'] + grp.get_group('Kansas').max()['cases']
  bordering_Nevada = grp.get_group('Idaho').max()['cases'] + grp.get_group('Oregon').max()['cases'] + grp.get_group('Utah').max()['cases'] + grp.get_group('Arizona').max()['cases'] + grp.get_group('California').max()['cases']
  bordering_NewHampshire = grp.get_group('Maine').max()['cases'] + grp.get_group('Vermont').max()['cases'] + grp.get_group('Massachusetts').max()['cases']
  bordering_NewJersey = grp.get_group('Pennsylvania').max()['cases'] + grp.get_group('Delaware').max()['cases'] + grp.get_group('New York').max()['cases']
  bordering_NewMexico = grp.get_group('Oklahoma').max()['cases'] + grp.get_group('Texas').max()['cases'] + grp.get_group('Utah').max()['cases'] + grp.get_group('Arizona').max()['cases'] + grp.get_group('Colorado').max()['cases']
  bordering_NewYork = grp.get_group('Pennsylvania').max()['cases'] + grp.get_group('Rhode Island').max()['cases'] + grp.get_group('Vermont').max()['cases'] + grp.get_group('Connecticut').max()['cases'] + grp.get_group('Massachusetts').max()['cases'] + grp.get_group('New Jersey').max()['cases']
  bordering_NorthCarolina = grp.get_group('Tennessee').max()['cases'] + grp.get_group('Virginia').max()['cases'] + grp.get_group('Georgia').max()['cases'] + grp.get_group('South Carolina').max()['cases']
  bordering_NorthDakota = grp.get_group('South Dakota').max()['cases'] + grp.get_group('Minnesota').max()['cases'] + grp.get_group('Montana').max()['cases']
  bordering_Ohio = grp.get_group('Michigan').max()['cases'] + grp.get_group('Pennsylvania').max()['cases'] + grp.get_group('West Virginia').max()['cases'] + grp.get_group('Kentucky').max()['cases'] + grp.get_group('Indiana').max()['cases']
  bordering_Oklahoma = grp.get_group('Missouri').max()['cases'] + grp.get_group('New Mexico').max()['cases'] + grp.get_group('Texas').max()['cases'] + grp.get_group('Arkansas').max()['cases'] + grp.get_group('Colorado').max()['cases'] + grp.get_group('Kansas').max()['cases']
  bordering_Oregon = grp.get_group('Nevada').max()['cases'] + grp.get_group('Washington').max()['cases'] + grp.get_group('California').max()['cases'] + grp.get_group('Idaho').max()['cases']
  bordering_Pennsylvania = grp.get_group('New York').max()['cases'] + grp.get_group('Ohio').max()['cases'] + grp.get_group('West Virginia').max()['cases'] + grp.get_group('Delaware').max()['cases'] + grp.get_group('New Jersey').max()['cases'] + grp.get_group('Maryland').max()['cases']
  bordering_RhodeIsland = grp.get_group('Massachusetts').max()['cases'] + grp.get_group('New York').max()['cases'] + grp.get_group('Connecticut').max()['cases']
  bordering_SouthCarolina = grp.get_group('North Carolina').max()['cases'] + grp.get_group('Georgia').max()['cases']
  bordering_SouthDakota = grp.get_group('Nebraska').max()['cases'] + grp.get_group('North Dakota').max()['cases'] + grp.get_group('Wyoming').max()['cases'] + grp.get_group('Iowa').max()['cases'] + grp.get_group('Minnesota').max()['cases'] + grp.get_group('Montana').max()['cases']
  bordering_Tennessee = grp.get_group('Mississippi').max()['cases'] + grp.get_group('Missouri').max()['cases'] + grp.get_group('North Carolina').max()['cases'] + grp.get_group('Virginia').max()['cases'] + grp.get_group('Alabama').max()['cases'] + grp.get_group('Arkansas').max()['cases'] + grp.get_group('Georgia').max()['cases'] + grp.get_group('Kentucky').max()['cases']
  bordering_Texas = grp.get_group('New Mexico').max()['cases'] + grp.get_group('Oklahoma').max()['cases'] + grp.get_group('Arkansas').max()['cases'] + grp.get_group('Louisiana').max()['cases']
  bordering_Utah = grp.get_group('Nevada').max()['cases'] + grp.get_group('New Mexico').max()['cases'] + grp.get_group('Wyoming').max()['cases'] + grp.get_group('Arizona').max()['cases'] + grp.get_group('Colorado').max()['cases'] + grp.get_group('Idaho').max()['cases']
  bordering_Vermont = grp.get_group('New Hampshire').max()['cases'] + grp.get_group('New York').max()['cases'] + grp.get_group('Massachusetts').max()['cases']
  bordering_Virginia = grp.get_group('North Carolina').max()['cases'] + grp.get_group('Tennessee').max()['cases'] + grp.get_group('West Virginia').max()['cases'] + grp.get_group('Kentucky').max()['cases'] + grp.get_group('Maryland').max()['cases']
  bordering_Washington = grp.get_group('Oregon').max()['cases'] + grp.get_group('Idaho').max()['cases']
  bordering_WestVirginia = grp.get_group('Pennsylvania').max()['cases'] + grp.get_group('Virginia').max()['cases'] + grp.get_group('Kentucky').max()['cases'] + grp.get_group('Maryland').max()['cases'] + grp.get_group('Ohio').max()['cases']
  bordering_Wisconsin = grp.get_group('Michigan').max()['cases'] + grp.get_group('Minnesota').max()['cases'] + grp.get_group('Illinois').max()['cases'] + grp.get_group('Iowa').max()['cases']
  bordering_Wyoming = grp.get_group('Nebraska').max()['cases'] + grp.get_group('South Dakota').max()['cases'] + grp.get_group('Utah').max()['cases'] + grp.get_group('Colorado').max()['cases'] + grp.get_group('Idaho').max()['cases'] + grp.get_group('Montana').max()['cases']


  ###############################################################################
  #creates a second graph with all the bordering states data
  ###############################################################################
  border = {'Border State': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'],
              'Total Cases': [bordering_alabama, bordering_alaska, bordering_arizona, bordering_Arkansas, bordering_California, bordering_Colorado, bordering_Connecticut, bordering_Delaware, bordering_florida, bordering_Georgia, bordering_Hawaii, bordering_Idaho, bordering_Illinois, bordering_Indiana, bordering_Iowa, bordering_Kansas, bordering_Kentucky, bordering_Louisiana, bordering_Maine, bordering_Maryland, bordering_Massachusetts, bordering_Michigan, bordering_Minnesota, bordering_Mississippi, bordering_Missouri, bordering_Montana, bordering_Nebraska, bordering_Nevada, bordering_NewHampshire, bordering_NewJersey, bordering_NewMexico, bordering_NewYork, bordering_NorthCarolina, bordering_NorthDakota, bordering_Ohio, bordering_Oklahoma, bordering_Oregon, bordering_Pennsylvania, bordering_RhodeIsland, bordering_SouthCarolina, bordering_SouthDakota, bordering_Tennessee, bordering_Texas, bordering_Utah, bordering_Vermont, bordering_Virginia, bordering_Washington, bordering_WestVirginia, bordering_Wisconsin, bordering_Wyoming]}
  border_df = pd.DataFrame(border)
  border_df.plot.bar(x="Border State", y="Total Cases", title="Bordering States Total Number of COVID19 Cases");
  plt.show(block=True)

  # #this is a test print to make sure the code is working for bordering states
  # print ("Total cases for Florida, Alabama and Georgia is: ", bordering_florida)

####################################################################################################
def question5(): ###################################################################################
####################################################################################################
#Could the rate of COVID be slowed in a highly controlled environment (state prison) vs a less controlled environment (states)?

  print()
  print('################')
  print("## Question 5 ##")
  print('################\n')
  print('Q. Could the rate of COVID be slowed in a highly controlled environment (state prison) vs a less controlled environment (states)?\n')

  file = 'us-states.csv'
  file2 = 'historical_state_counts.csv'

  # df = pd.read_csv(file)
  # state_grouped = df.groupby('state', sort = 'True')
  # florida_grouped = state_grouped.get_group('Florida')
  # print(florida_grouped)

  ############################################################
  #Prison cases df
  ############################################################

  prison_df = pd.read_csv(file2)
  #print(list(prison_df.columns))
  state_death_df = prison_df[['Date','State','Residents.Confirmed','Residents.Deaths']]
  prison_grouped = state_death_df.groupby('State',sort='True')
  Maryland_grouped = prison_grouped.get_group('Maryland')
  #print(Maryland_grouped[['Date','State','Residents.Confirmed']])
  Maryland_date_residents = Maryland_grouped[['Date','State','Residents.Confirmed']]
  Maryland_date_residents = Maryland_date_residents.fillna(0)
  #print(Maryland_date_residents)
  Maryland_date_residents.plot(x='Date',y='Residents.Confirmed')
  #plt.show()
  #Maryland_confirmed = prison_grouped.get_group('Maryland').max()['Residents.Confirmed']



  ############################################################
  #State cases df
  ############################################################

  df = pd.read_csv(file)
  grouped = df.groupby('state')
  state_Maryland = grouped.get_group('Maryland')
  #state_Maryland.plot(x="date", y="cases", title="Number of total COVID-19 cases");
  #plt.show()


  Maryland_residents = Maryland_grouped[['Date','Residents.Confirmed']]
  state_Maryland_cases = state_Maryland[['date','cases']]
  state_Maryland['date'] = pd.to_datetime(state_Maryland['date'])
  print(state_Maryland.info())
  print(type(state_Maryland['date']))

  Maryland_grouped['Date'] = pd.to_datetime(Maryland_grouped['Date'])
  print(type(Maryland_grouped.info()))
  print(state_Maryland_cases)

  #ax = Maryland_residents.plot(x = 'Date', y = 'Residents.Confirmed')
  #state_Maryland_cases.plot(ax=ax, x ='date', y = 'cases')
  #plt.show()



  #############################################################
  #CreateFigure
  #############################################################

  fig = plt.figure(figsize = (12,6))


  ############################################################
  #Create Axes
  ############################################################
  ax = fig.add_subplot(111, label = '1')
  ax2 = fig.add_subplot(111, label = '2', frame_on = False)

  #########################################################
  #Attempting to fix the tick marks -- Not working
  #########################################################
  #x = np.random.randint(low=0, high=1, size=50)
  #y = np.random.randint(low=0, high=1, size=50)
  # ax.set_xticks(np.arange(0, len(x)+1, 12))
  # ax.set_yticks(np.arange(0, max(y), 48))
  # ax2.set_xticks(np.arange(0, len(x)+1, 12))
  # ax2.set_yticks(np.arange(0, max(y), 48))

  ############################################################
  #Set parameter first Axes
  ############################################################
  ax.bar(x = Maryland_grouped['Date'], height = Maryland_grouped['Residents.Confirmed'], color = 'C1')
  ax.set_xlabel('Date', color = 'C1')
  ax.set_ylabel('New Coronavirus Cases - State', color = 'C1')
  ax.yaxis.tick_right()
  ax.yaxis.set_label_position('right')
  ax.tick_params(axis = 'x', color = 'C1')
  ax.tick_params(axis = 'y', color = 'C1')
  plt.xticks(rotation=90)

  ############################################################
  #Set parameter second Axes
  ############################################################
  ax2.plot(state_Maryland['date'], state_Maryland['cases'], color = 'C0')
  ax2.set_xlabel('Date', color = 'C0')
  ax2.set_ylabel('New Coronavirus Cases - Prison', color = 'C0')
  ax2.xaxis.tick_top()
  ax2.xaxis.set_label_position('top')
  #plt.xticks(rotation=90)

  ############################################################
  #Display
  ############################################################
  plt.tight_layout()
  plt.show()

  #this group simply groups each states
  #print(df)

  # #this grouped returns the max covid cases for each State
  # grouped = df.groupby('state', sort=True).max().reset_index()

####################################################################################################
def question6(): ###################################################################################
####################################################################################################
#How much of an issue was COVID-19 in the last presedential debate series?

  print()
  print('################')
  print("## Question 6 ##")
  print('################\n')
  print('Q. How much of an issue was COVID-19 in the last presedential debate series?\n')

  source = requests.get('http://www.debates.org/index.php?page=debate-transcripts').content

  soup = bs.BeautifulSoup(source,'html.parser')

  content = soup.find(id='content-sm')

  theDebate=[]

  for link in content.findAll('a'):
    if '2020' in link.string:
      theDebate.append(link.get('href'))

  print(theDebate)

  for link in content.findAll('a'):
    if '2020' in link.string:
      print(link.get('href'))

  title=[]

  for link in content.findAll('a'):
    if '2020' in link.string:
      title.append(link.string)

  print(title)

  df = pd.DataFrame(columns=title)

  print(df)


  nospacechars = []

  for i in theDebate:
    source = requests.get(i).content
    soup = bs.BeautifulSoup(source,'html.parser')
    content = soup.find(id='content-sm')
    count = content.find('p').text
    count = count.replace('\n', '')
    nospacechars.append(len(re.sub(r"\s+", "", count)))

  print(nospacechars)


  ############################################################################
  #Count how many times covid of pandemic comes up
  ############################################################################

  covid_count=[]

  for i in theDebate:
    source = requests.get(i).content
    soup = bs.BeautifulSoup(source,'html.parser')
    content = soup.find(id='content-sm').text
    #count = content.find('p').text
    #a = re.split(r'\w', count)
    #b = a.count('covid')+a.count('pandemic')+a.count('Covid19')+a.count('COVID-19')
    #covid_count.append(b)
    final = content.count('COVID-19') + content.count('COVID') + content.count('coronavirus') + content.count('virus') + content.count('pandemic')
    covid_count.append(final)

  print(covid_count)


  ############################################################################
  #Setup the dataframes
  ############################################################################


  df.loc[2] = covid_count


  df['Name']=['covid_count']

  df.set_index('Name')

  pd.options.display.max_colwidth = 10
  print(df.set_index('Name'))


  def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(15, 10))
    # Display image
    plt.imshow(wordcloud, interpolation = 'bilinear')
    # No axis details
    plt.axis("off");
    plt.show()

  # Import image to np.array
  #mask = np.array(Image.open('stormtrooper_mask.png'))

  # Generate wordcloud
  stop_words = ["Trump", "Biden", "re", 's', 't', 'welker'] + list(STOPWORDS)
  wordcloud = WordCloud(width = 2000, height = 1000, random_state=1, background_color='black', colormap='rainbow', collocations=False, stopwords = stop_words).generate(content)

  # Plot
  plot_cloud(wordcloud)


####################################################################################################
def question7(): ###################################################################################
####################################################################################################
#As vaccinations continue to be administered to the general public, have the numbers of infected unvaccinated people gone down or up?

  print()
  print('################')
  print("## Question 7 ##")
  print('################\n')
  print('Q. As vaccinations continue to be administered to the general public, have the numbers of infected unvaccinated people gone down or up?\n')

  df = pd.read_csv('dc_breakthrough_11_22_2021.csv')
  df = df.rename(columns={'Unnamed: 0': ''})
  display(df)

  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.axis('equal')
  labels = list()
  for item in df[''].iloc[:3]:
    labels.append(item)
  data = list()
  for item in df['Percent of cases since Jan 18th 2021'].iloc[:3]:
    data.append(item)
  ax.pie(data, labels = labels, autopct='%1.1f%%')
  plt.title('Percent of positive COVID cases from Jan 18th 2021 - Nov 22 2021')
  plt.show()

  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.axis('equal')
  labels = list()
  for item in df[''].iloc[:3]:
    labels.append(item)
  data = list()
  for item in df['Percent of cases in last 7 days'].iloc[:3]:
    data.append(item)
  ax.pie(data, labels = labels, autopct='%1.1f%%')
  plt.title('Percent of positive COVID cases from Nov 15th 2021 - Nov 22 2021')
  plt.show()

  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.axis('equal')
  labels = list()
  for item in df[''].iloc[:3]:
    labels.append(item)
  data = list()
  for item in df['Percent of cases since Jan 18th 2021'].iloc[:3]:
    data.append(item)
  ax.pie(data, labels = labels, autopct='%1.1f%%')
  plt.title('Percent of positive COVID cases from Jan 18th 2021 - Nov 22 2021')
  plt.show()

  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.axis('equal')
  labels = ['Fully Vaccinated', 'Not Fully Vaccinated']
  data = list()
  data.append(df['Percent of cases in last 7 days'].iloc[0])
  data.append(df['Percent of cases in last 7 days'].iloc[1]+df['Percent of cases in last 7 days'].iloc[2])
  ax.pie(data, labels = labels, autopct='%1.1f%%')
  plt.title('Percent of positive COVID cases from Nov 15th 2021 - Nov 22 2021')
  plt.show()

  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.axis('equal')
  labels = ['Fully Vaccinated', 'Not Fully Vaccinated']
  data = list()
  data.append(df['Percent of cases since Jan 18th 2021'].iloc[0])
  data.append(df['Percent of cases since Jan 18th 2021'].iloc[1]+df['Percent of cases since Jan 18th 2021'].iloc[2])
  ax.pie(data, labels = labels, autopct='%1.1f%%')
  plt.title('Percent of positive COVID cases from Jan 18th 2021 - Nov 22 2021')
  plt.show()

####################################################################################################
def question8(): ###################################################################################
####################################################################################################
#What are the demographics of the people who visit the places where infection outbreaks occur?

  print()
  print('################')
  print("## Question 8 ##")
  print('################\n')
  print('Q. What are the demographics of the people who visit the places where infection outbreaks occur?\n')

  df = pd.read_csv('DC_COVID-19_Outbreaks.csv')
  df.dropna(how='any')
  setting = df['SETTING_TYPE'].unique()
  new = pd.DataFrame(columns= ['SETTING_TYPE', 'TOTAL_OUTBREAKS'])
  new.reset_index(drop=True)
  x = list()
  for item in setting:
    x.append(item)

  new['SETTING_TYPE'] = x
  total = list()
  for item in setting:
    total.append(int(df.loc[df['SETTING_TYPE'] == item][['OUTBREAKS_NUMBER']].sum()))

  new['TOTAL_OUTBREAKS'] = total
  display(new.sort_values(by = 'TOTAL_OUTBREAKS', ascending=False)) #####

  print(df['WEEK_ENDING_DATE'].min()) #####

  print(df['WEEK_ENDING_DATE'].max()) #####

  cloud = ''
  for item in df['SETTING_TYPE'].unique():
    cloud+= item + ' '
  cloud

  #def plot_cloud(wordcloud):
  #  # Set figure size
  #  plt.figure(figsize=(20, 10))
  #  # Display image
  #  plt.imshow(wordcloud)
  #  # No axis details
  #  plt.axis("off")
  wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='GnBu_r', collocations=False, stopwords = STOPWORDS).generate(cloud)
  #plot_cloud(wordcloud)

  # Set figure size
  plt.figure(figsize=(20, 10))
  # Display image
  plt.imshow(wordcloud)
  # No axis details
  plt.axis("off")
  plt.show() #####

####################################################################################################
def question9(): ###################################################################################
####################################################################################################
#Are there any racial disparities in DC when it comes to COVID data?

  print()
  print('################')
  print("## Question 9 ##")
  print('################\n')
  print('Q. Are there any racial disparities in DC when it comes to COVID data?\n')

  # Open the files and add wards to dataframe
  unemployment = pd.read_csv('chart.csv')
  education = pd.read_csv('Population_Education.csv')
  total_tests = pd.read_csv('Total_Tests_By_Ward.csv', encoding = 'UTF-16')
  total_positives = pd.read_csv('Total_Positives_By_Ward.csv', encoding = 'UTF-16')
  total_lives_lost = pd.read_csv('Total_Lives_Lost_By_Ward.csv', encoding = 'UTF-16')
  avg_income_dc = pd.read_csv('Average_Household_Income_by_Race_Ethnicity_City_District_of_Columbia.csv')
  avg_income_ward1 = pd.read_csv('Average_Household_Income_by_Race_Ethnicity_Ward_Ward_1.csv')
  avg_income_ward1['Ward'] = 'Ward 1'
  avg_income_ward2 = pd.read_csv('Average_Household_Income_by_Race_Ethnicity_Ward_Ward_2.csv')
  avg_income_ward2['Ward'] = 'Ward 2'
  avg_income_ward3 = pd.read_csv('Average_Household_Income_by_Race_Ethnicity_Ward_Ward_3.csv')
  avg_income_ward3['Ward'] = 'Ward 3'
  avg_income_ward4 = pd.read_csv('Average_Household_Income_by_Race_Ethnicity_Ward_Ward_4.csv')
  avg_income_ward4['Ward'] = 'Ward 4'
  avg_income_ward5 = pd.read_csv('Average_Household_Income_by_Race_Ethnicity_Ward_Ward_5.csv')
  avg_income_ward5['Ward'] = 'Ward 5'
  avg_income_ward6 = pd.read_csv('Average_Household_Income_by_Race_Ethnicity_Ward_Ward_6.csv')
  avg_income_ward6['Ward'] = 'Ward 6'
  avg_income_ward7 = pd.read_csv('Average_Household_Income_by_Race_Ethnicity_Ward_Ward_7.csv')
  avg_income_ward7['Ward'] = 'Ward 7'
  avg_income_ward8 = pd.read_csv('Average_Household_Income_by_Race_Ethnicity_Ward_Ward_8.csv')
  avg_income_ward8['Ward'] = 'Ward 8'
  race_dc = pd.read_csv('Population_by_Race_City_District_of_Columbia.csv')
  race_ward1 = pd.read_csv('Population_by_Race_Ward_Ward_1.csv')
  race_ward1['Ward'] = 'Ward 1'
  race_ward2 = pd.read_csv('Population_by_Race_Ward_Ward_2.csv')
  race_ward2['Ward'] = 'Ward 2'
  race_ward3 = pd.read_csv('Population_by_Race_Ward_Ward_3.csv')
  race_ward3['Ward'] = 'Ward 3'
  race_ward4 = pd.read_csv('Population_by_Race_Ward_Ward_4.csv')
  race_ward4['Ward'] = 'Ward 4'
  race_ward5 = pd.read_csv('Population_by_Race_Ward_Ward_5.csv')
  race_ward5['Ward'] = 'Ward 5'
  race_ward6 = pd.read_csv('Population_by_Race_Ward_Ward_6.csv')
  race_ward6['Ward'] = 'Ward 6'
  race_ward7 = pd.read_csv('Population_by_Race_Ward_Ward_7.csv')
  race_ward7['Ward'] = 'Ward 7'
  race_ward8 = pd.read_csv('Population_by_Race_Ward_Ward_8.csv')
  race_ward8['Ward'] = 'Ward 8'

  # Sort by ward
  unemployment = unemployment.sort_values(by='Ward')
  education = education.rename(columns={'Population Age 25+ with Less Than High School Graduation\n': "Population Age 25+ with Less Than High School Graduation"})
  education = education.sort_values(by='Ward')

  # Add all ward average income data into 1 dataframe
  avg_income = pd.DataFrame()
  avg_income = avg_income.append(avg_income_ward1)
  avg_income = avg_income.append(avg_income_ward2)
  avg_income = avg_income.append(avg_income_ward3)
  avg_income = avg_income.append(avg_income_ward4)
  avg_income = avg_income.append(avg_income_ward5)
  avg_income = avg_income.append(avg_income_ward6)
  avg_income = avg_income.append(avg_income_ward7)
  avg_income = avg_income.append(avg_income_ward8)
  avg_income = avg_income.rename(columns={'Category': "Race"})
  avg_income = avg_income.drop(columns=['Average Household Income by Race/Ethnicity (Dollars) .1'])
  avg_income.drop(avg_income[avg_income['Race'] == 'Non-Hispanic/Latino'].index, inplace = True)
  avg_income.drop(avg_income[avg_income['Race'] == 'Hispanic/Latino'].index, inplace = True)

  # Add all racial population data into 1 dataframe
  race = pd.DataFrame()
  race = race.append(race_ward1)
  race = race.append(race_ward2)
  race = race.append(race_ward3)
  race = race.append(race_ward4)
  race = race.append(race_ward5)
  race = race.append(race_ward6)
  race = race.append(race_ward7)
  race = race.append(race_ward8)
  race = race.rename(columns={'Category': "Race"})

  # Display a graph showing the distribution of race within each ward
  fig = px.bar(race, x='Ward', y='Population by Race', color='Race')
  fig.show()

  # Display average income based on race for each ward
  fig = px.bar(avg_income.drop(avg_income[avg_income['Race'] == 'All'].index, inplace = False), x='Ward', y='Average Household Income by Race/Ethnicity (Dollars) ', color='Race')
  fig.show()

  whitepd = pd.DataFrame(race[race['Race']=='White'])
  blackpd = pd.DataFrame(race[race['Race']=='Black/African American'])
  allincome = pd.DataFrame(avg_income[avg_income['Race']=='All'])
  allincome = allincome.sort_values(by='Ward')
  white = go.Scatter(name='Total White Population', x=whitepd['Ward'], y=whitepd['Population by Race'])
  income = go.Scatter(name='Average Income of Ward', x=whitepd['Ward'], y=allincome['Average Household Income by Race/Ethnicity (Dollars) '])
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(white,secondary_y=True)
  fig.add_trace(income)
  fig['layout'].update(height = 600, width = 1000, title = 'Total White Population vs Average Income by Ward')
  fig.update_yaxes(title_text="Total White Population", secondary_y=False)
  fig.update_yaxes(title_text="Average Income of Ward", secondary_y=True)
  fig.show()

  whitepd = pd.DataFrame(race[race['Race']=='White'])
  blackpd = pd.DataFrame(race[race['Race']=='Black/African American'])
  allincome = pd.DataFrame(avg_income[avg_income['Race']=='All'])
  allincome = allincome.sort_values(by='Ward')
  black = go.Scatter(name='Total Black/African American Population', x=blackpd['Ward'], y=blackpd['Population by Race'])
  income = go.Scatter(name='Average Income of Ward', x=blackpd['Ward'], y=allincome['Average Household Income by Race/Ethnicity (Dollars) '])
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(black,secondary_y=True)
  fig.add_trace(income)
  fig['layout'].update(height = 600, width = 1000, title = 'Total Black/African American Population vs Average Income by Ward')
  fig.update_yaxes(title_text="Total Black/African American Population", secondary_y=False)
  fig.update_yaxes(title_text="Average Income of Ward", secondary_y=True)
  fig.show()

  edu = go.Scatter(name='Population with Less Than High School Graduation (%)', x=education['Ward'], y=education['Population Age 25+ with Less Than High School Graduation'])
  white = go.Scatter(name='Total White Population', x=education['Ward'], y=whitepd['Population by Race'])
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(white,secondary_y=True)
  fig.add_trace(edu)
  fig['layout'].update(height = 600, width = 1000, title = 'Population with Less Than High School Graduation (%) vs. Total White  Population')
  fig.update_yaxes(title_text="Population with Less Than High School Graduation (%)", secondary_y=False)
  fig.update_yaxes(title_text="Total White Population", secondary_y=True)
  fig.show()

  edu = go.Scatter(name='Population with Less Than High School Graduation (%)', x=education['Ward'], y=education['Population Age 25+ with Less Than High School Graduation'])
  black = go.Scatter(name='Total Black/African American Population', x=education['Ward'], y=blackpd['Population by Race'])
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(black,secondary_y=True)
  fig.add_trace(edu)
  fig['layout'].update(height = 600, width = 1000, title = 'Population with Less Than High School Graduation (%) vs. Total Black/African American Population')
  fig.update_yaxes(title_text="Population with Less Than High School Graduation (%)", secondary_y=False)
  fig.update_yaxes(title_text="Total Black/African American Population", secondary_y=True)
  fig.show()

  unemp = go.Scatter(name='Unemployment Percentage', x=whitepd['Ward'], y=unemployment['Unemployment Percentage'])
  black = go.Scatter(name='Total Black/African American Population', x=whitepd['Ward'], y=blackpd['Population by Race'])
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(white,secondary_y=True)
  fig.add_trace(unemp)
  fig['layout'].update(height = 600, width = 1000, title = 'Unemployment Percentage vs. Total White Population')
  fig.update_yaxes(title_text="Unemployment Percentage", secondary_y=False)
  fig.update_yaxes(title_text="Total White Population", secondary_y=True)
  fig.show()

  unemp = go.Scatter(name='Unemployment Percentage', x=whitepd['Ward'], y=unemployment['Unemployment Percentage'])
  black = go.Scatter(name='Total Black/African American Population', x=whitepd['Ward'], y=blackpd['Population by Race'])
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(black,secondary_y=True)
  fig.add_trace(unemp)
  fig['layout'].update(height = 600, width = 1000, title = 'Unemployment Percentage vs. Total Black/African American Population')
  fig.update_yaxes(title_text="Unemployment Percentage", secondary_y=False)
  fig.update_yaxes(title_text="Total Black/African American Population", secondary_y=True)
  fig.show()

  url = 'https://em.dcgis.dc.gov/dcgis/rest/services/COVID_19/OpenData_COVID19/FeatureServer/48/query?where=1%3D1&outFields=WARD,COVERAGE_65_PLUS,COVERAGE_ALL,VACCINATED_65_PLUS,VACCINATED_ALL&outSR=4326&f=json'
  #uh = urllib.request.urlopen(url)
  uh = ul.urlopen(url)

  data = uh.read()
  data = json.loads(data)
  vaccine = pd.DataFrame()
  for x in range(0,8):
    vaccine = vaccine.append(data['features'][x]['attributes'], ignore_index=True)
  vaccine = vaccine.rename(columns={'WARD': "Ward", 'COVERAGE_ALL' : 'Total Coverage', 'VACCINATED_ALL' : 'Total Vaccinated'})

  total_positives.drop(total_positives[total_positives['Ward'] == 'Unknown'].index, inplace = True)
  total_positives = total_positives.sort_values(by='Ward')
  vaccine = vaccine.sort_values(by='Ward')

  fig = make_subplots(rows=1, cols=2)

  fig.add_trace(
      go.Scatter(name='Total Vaccinated', x=total_positives['Ward'], y=vaccine['Total Vaccinated']),
      row=1, col=1
  )

  fig.add_trace(go.Scatter(name='Total Positives', x=total_positives['Ward'], y=total_positives['Total Positives']), row=1, col=2)

  fig.update_layout(height=600, width=800, title_text="Total Vaccinated vs Total Positives by Ward")
  fig.show()

  trace1 = go.Scatter(name='Total Positives', x=total_positives['Ward'], y=total_positives['Total Positives'])
  trace2 = go.Scatter(name='Total Vaccinated', x=total_positives['Ward'], y=vaccine['Total Vaccinated'])
  total_positives["Total Positives"]=total_positives["Total Positives"].astype("string")
  total_positives["Total Positives"] = total_positives["Total Positives"].apply(lambda x: x.replace(",","")).astype(float)
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(trace1)
  fig.add_trace(trace2,secondary_y=True)
  fig['layout'].update(height = 600, width = 800, title = 'Total Vaccinated vs Total Positives')
  fig.update_yaxes(title_text="Total Positives", secondary_y=False)
  fig.update_yaxes(title_text="Total Vaccinated", secondary_y=True)
  fig.show()

  total_tests.drop(total_tests[total_tests['Ward'] == 'Unknown'].index, inplace = True)
  total_tests = total_tests.sort_values(by='Ward')
  total_tests["Total Tests"]=total_tests["Total Tests"].astype("string")
  total_tests["Total Tests"] = total_tests["Total Tests"].apply(lambda x: x.replace(",","")).astype(float)
  fig = make_subplots(rows=1, cols=2)

  fig.add_trace(go.Scatter(name='Total Tests', x=total_tests['Ward'], y=total_tests['Total Tests']),row=1, col=1)

  fig.add_trace(go.Scatter(name='Total Positives', x=total_tests['Ward'], y=total_positives['Total Positives']), row=1, col=2)

  fig.update_layout(height=600, width=800, title_text="Total Tests vs Total Positives by Ward")
  fig.show()

  trace1 = go.Scatter(name='Total Positives', x=total_positives['Ward'], y=total_positives['Total Positives'])
  trace2 = go.Scatter(name='Total Tests', x=total_positives['Ward'], y=total_tests['Total Tests'])

  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(trace1)
  fig.add_trace(trace2,secondary_y=True)
  fig['layout'].update(height = 600, width = 1000, title = 'Total Tests vs Total Positives')
  fig.update_yaxes(title_text="Total Positives", secondary_y=False)
  fig.update_yaxes(title_text="Total Tests", secondary_y=True)
  fig.show()

  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(trace1,secondary_y=True)
  fig.add_trace(black)
  fig['layout'].update(height = 600, width = 1000, title = 'Total Black/African American Population vs Total Positives')
  fig.update_yaxes(title_text="Total Black/African American Population", secondary_y=False)
  fig.update_yaxes(title_text="Total Positives", secondary_y=True)
  fig.show()

  total_lives_lost.drop(total_lives_lost[total_lives_lost['Ward'] == 'Unknown'].index, inplace = True)
  total_lives_lost.drop(total_lives_lost[total_lives_lost['Ward'] == 'Homeless'].index, inplace = True)

  total_lives_lost = total_lives_lost.sort_values(by='Ward')
  lost = go.Scatter(name = 'Total Lives Lost', x=total_positives['Ward'],y=total_lives_lost['Total Lives Lost'])
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(black)
  fig.add_trace(lost,secondary_y=True)
  fig['layout'].update(height = 600, width = 1000, title = 'Total Black/African American Population vs Total Lives Lost')
  fig.update_yaxes(title_text="Total Black/African American Population", secondary_y=False)
  fig.update_yaxes(title_text="Total Lives Lost", secondary_y=True)
  fig.show()

####################################################################################################
def question10(): ##################################################################################
####################################################################################################
#What relationhisp is there between Covid-19 infections and unemployment claims?
#Data
#https://beta.bls.gov/dataViewer/view/timeseries/LASST110000000000004 -- unemployment
#https://healthdata.gov/dataset/United-States-COVID-19-Cases-and-Deaths-by-State-o/hiyb-zgc2 -- covid cases

  ##################################################
  ## Unemployment df ###############################
  ##################################################

  ##################################################
  #Create unemployment df
  try:
    df_dc_unem = pd.read_csv('dc_unemployment_claims.csv')
    #print('df created')
  except:
    print('File not found')
    exit(-1)

  ##################################################
  #Drop unwanted column
  df_dc_unem = df_dc_unem.drop(['Series ID'], axis = 1)

  ##################################################
  #Drop data before Jan 2020
  selector = df_dc_unem[df_dc_unem['Year'] != 2019]
  df_dc_unem = selector

  ##################################################
  ## Cases df ######################################
  ##################################################

  ##################################################
  #Create cases df
  try:
    df_cases = pd.read_csv('covid_cases.csv')
  except:
    print('File not found')
    exit(-1)

  ##################################################
  #select for desired columns
  selector = df_cases[['submission_date', 'state', 'new_case', 'new_death']]

  ##################################################
  #Filter for state == DC
  df_dc_cases = selector[selector['state'] == 'DC'].copy() # -- had to explicity make a copy here to avoid SettingWithCopy warning

  ##################################################
  #Convert dates to datetime variable and sort
  df_dc_cases['submission_date'] = pd.to_datetime(df_dc_cases['submission_date']) #--If I did not make an explicit copy() above, the SettingWithCopy warning happened here
  df_dc_cases = df_dc_cases.sort_values(by = 'submission_date', ascending = True)

  ##################################################
  #Drop data after Sep 2021
  selector = df_dc_cases[df_dc_cases['submission_date'] < ('2021-10-01')]
  df_dc_cases = selector

  ##################################################
  ## Commentary ####################################
  ##################################################
  print()
  print('#################')
  print("## Question 10 ##")
  print('#################\n')
  print('Q. What relationhisp is there between Covid-19 infections and unemployment claims?\n')
  print('By observing this graph, it can be seen that the rise in unemployment claims\n\
coincided with the first wave of Covid-19 infections in March-May 2020.\n\
Interestingly, the unemployment claims did not have a similar correlation with\n\
the second and third waves of infection. Rather they began to steadily decline in Jan 2021\n\
irrespective of Covid-19 infections.\n')

  ##################################################
  ## Graph data ####################################
  ##################################################

  ##################################################
  #Create figure
  fig = plt.figure(figsize = (12, 6))

  ##################################################
  #Create axes
  ax = fig.add_subplot(111, label = '1')
  ax2 = fig.add_subplot(111, label = '2', frame_on = False)

  ##################################################
  #Set parameters first axis
  ax.bar(x = df_dc_cases['submission_date'], height = df_dc_cases['new_case'], color = 'olive')
  ax.set_xlabel('Date')
  ax.set_ylabel('New Coronavirus Cases', color = 'olive')
  ax.set_title('Correlation of Covid First Wave with Unemployment Claims')
  ax.yaxis.tick_right()
  ax.yaxis.set_label_position('right')
  ax.tick_params(axis = 'y', color = 'olive')

  ##################################################
  #Set parameters second axis
  ax2.plot(df_dc_unem['Label'], df_dc_unem['Value'], color = 'brown')
  ax2.set_ylabel('Monthly Unemployment Claims', color = 'brown')
  ax2.tick_params(axis = 'y', color = 'brown')
  ax2.axes.xaxis.set_visible(False)

  ##################################################
  #Display
  plt.tight_layout() #makes graph fit scale
  plt.show()

####################################################################################################
def question11(): ##################################################################################
####################################################################################################
#What relationhsip is there between Covid-19 vaccination rates and unemployment claims?
#Data
#https://beta.bls.gov/dataViewer/view/timeseries/LASST110000000000004 -- unemployment
#https://data.cdc.gov/resource/unsk-b7fc.json?location=DC -- vaccinations

  ##################################################
  ## Vaccination df ################################
  ##################################################

  #Create df with CDC API
  api_url = 'https://data.cdc.gov/resource/unsk-b7fc.json?location=DC' #note, this API already filters for location

  try:
    data = json.load(ul.urlopen(api_url))
    df_vac = json_normalize(data)
  except:
    print('data not found')
    exit(-1)

  ##################################################
  #Select desired columns
  selector = df_vac[['date', 'location', 'administered_dose1_pop_pct', 'series_complete_pop_pct']]
  df_vac_focused = selector.copy()

  ##################################################
  #Convert dates to datetime and percentages to float, then sort
  df_vac_focused['date'] = pd.to_datetime(df_vac_focused['date'])
  df_vac_focused['administered_dose1_pop_pct'] = df_vac_focused['administered_dose1_pop_pct'].astype(str).astype(np.float64)
  df_vac_focused['series_complete_pop_pct'] = df_vac_focused['series_complete_pop_pct'].astype(str).astype(np.float64)
  df_vac_focused = df_vac_focused.sort_values(by = 'date', ascending = True)

  ##################################################
  #Drop data after Sep 2021
  selector = df_vac_focused[df_vac_focused['date'] < ('2021-10-01')]
  df_vac_focused = selector.copy()

  #print(df_vac_focused)

  ##################################################
  ## Unemployment df ###############################
  ##################################################

  #Create unemployment df
  try:
    df_dc_unem = pd.read_csv('dc_unemployment_claims.csv')
  except:
    print('File not found')
    exit(-1)

  ##################################################
  #Drop unwanted column
  df_dc_unem = df_dc_unem.drop(['Series ID'], axis = 1)

  ##################################################
  #Drop data before Jan 2021
  selector = df_dc_unem[(df_dc_unem['Year'] >= 2021)]
  df_dc_unem = selector

  ##################################################
  ## Commentary ####################################
  ##################################################
  print()
  print('#################')
  print("## Question 11 ##")
  print('#################\n')
  print('Q. What relationhsip is there between Covid-19 vaccination rates and unemployment claims?\n')
  print('This graph gives some insight into what led to a steady decrease in unemployment claims\n\
beginning in Jan 2021 despite there being a second and third wave of Covid-19 infections.\n\
The vaccine began to become available at this time. As the percentage of vaccinated individuals grew,\n\
the unemployment claims declined. This continued to hold true despite additional waves of infection.\n')

  ##################################################
  ## Graph data ####################################
  ##################################################

  ##################################################
  #Create figure
  fig = plt.figure(figsize = (12, 6))

  ##################################################
  #Create axes
  ax = fig.add_subplot(111, label = '1')
  ax2 = fig.add_subplot(111, label = '2', frame_on = False)

  #Set parameters first axis
  ax.bar(x=df_vac_focused['date'], height = df_vac_focused['administered_dose1_pop_pct'], color = 'lightsteelblue')
  ax.set_title('Correlation of Vaccinations with Drop in Unemployment Claims')
  ax.set_xlabel('Date')
  ax.set_ylabel('Cumulative 1-Dose Vaccination Rates in DC (%)', color = 'lightsteelblue')
  ax.yaxis.tick_right()
  ax.tick_params(axis = 'y', color = 'lightsteelblue')
  ax.yaxis.set_label_position('right')

  ##################################################
  #Set parameters second axis
  ax2.plot(df_dc_unem['Label'], df_dc_unem['Value'], color = 'brown')
  ax2.set_ylabel('Monthly Unemployment Claims', color = 'brown')
  ax2.tick_params(axis = 'y', color = 'brown')
  ax2.axes.xaxis.set_visible(False)

  ##################################################
  #Display
  plt.tight_layout() #makes graph fit scale
  plt.show()

####################################################################################################
def question12(): ##################################################################################
####################################################################################################
#How significant of an impact did Covid-19 have on unemployment claims?

  ##################################################
  ## manually work with data #######################
  ##################################################

  ##################################################
  #Open and Read csv file
  year =list()
  month = list()
  value = list()
  per_incr = list()

  with open('dc_unemployment_claims.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
      #print(row)
      year.append(row[1])
      month.append(row[2])
      value.append(row[4])


  ##################################################
  #drop column names
  year.pop(0)
  month.pop(0)
  value.pop(0)

  ##################################################
  #Change month to numeric
  n = 0
  for element in month:
    if element == 'M01':
      month[n] = 1
    if element == 'M02':
      month[n] = 2
    if element == 'M03':
      month[n] = '3'
    if element == 'M04':
      month[n] = 4
    if element == 'M05':
      month[n] = 5
    if element == 'M06':
      month[n] = 6
    if element == 'M07':
      month[n] = 7
    if element == 'M08':
      month[n] = 8
    if element == 'M09':
      month[n] = 9
    if element == 'M10':
      month[n] = 10
    if element == 'M11':
      month[n] = 11
    if element == 'M12':
      month[n] = 12
    n += 1

  #print(year)
  #print(month)
  #print(value)

  ##################################################
  #Calculate percent increase in unemployment claims month over month
  n=0
  for element in value:
    if n == 0:
      per_incr.append(0)
    else:
      change = float(element) / float(value[n - 1]) - 1
      #print(element, value[n], change)
      per_incr.append(change)
    n += 1

  ##################################################
  ## monthly change df #############################
  ##################################################

  ##################################################
  #Create monthly change df
  unem_change_data = { 'Year': year, 'Month': month, 'Value': value, 'Per_Incr': per_incr }
  df_monthly_change = pd.DataFrame(unem_change_data)

  ##################################################
  #Convert columns to appropriate data types
  df_monthly_change['Year'] = df_monthly_change['Year'].astype(np.int64)
  df_monthly_change['Month'] = df_monthly_change['Month'].astype(np.int64)
  df_monthly_change['Value'] = df_monthly_change['Value'].astype(np.int64)
  df_monthly_change['Per_Incr'] = df_monthly_change['Per_Incr'].astype(np.float64)

  ##################################################
  #Create a date column by combining Year and Month, convert to datetime and sort
  df_monthly_change['Date'] = df_monthly_change['Year'].astype(str) + '-' + df_monthly_change['Month'].astype(str) + '-01'
  df_monthly_change['Date'] = pd.to_datetime(df_monthly_change['Date']).dt.date #dt.date gets rid of hours/sec
  df_monthly_change = df_monthly_change.sort_values(by = 'Date', ascending = True)

  ##################################################
  #Create a % increase column with pandas
  #df_monthly_change['Per_Increase'] = df_monthly_change['Value'].pct_change()

  ##################################################
  #Reorder columns dropping unwanted ones, i.e. drops individual columns for month and year, leaving only the new date column
  df_monthly_change = df_monthly_change.reindex(columns=['Date', 'Value', 'Per_Incr'])

  ##################################################
  #Display df
  #print(df_monthly_change)

  ##################################################
  ## Graph data ####################################
  ##################################################

  ##################################################
  #Set parameters of monthly change graph
  colors = list()
  for element in df_monthly_change['Per_Incr']:
    if element > 0.0:
      colors.append('brown')
    else:
      colors.append('thistle')

  df_monthly_change.plot(kind = 'bar', x = 'Date', y = 'Per_Incr', figsize=(12, 6), color = colors)
  plt.title('Monthly Percentage Change of Unemployment Claims')
  plt.xlabel('Date')
  plt.ylabel('Change in Unemployment Claims (%)')

  ##################################################
  #Commentary
  print()
  print('#################')
  print("## Question 12 ##")
  print('#################\n')
  print('Q. How significant of an impact did Covid-19 have on unemployment claims?\n')
  print('There was over a 100 percent increase in the unemplyment claims from Mar 2020 to April 2020.\n\
The graph shows just how unusual this was, signifiying the extreme impact\n\
the arrival of Covid had on increased unemployment claims. Although not nearly\n\
as significant as the arrival of Covid, one can also see that the arrival of the\n\
vaccine in Jan 2021 also had an outsized impact on decreased unemployment claims.\n')

  ##################################################
  #Display monthly change graph
  plt.tight_layout() #makes graph fit scale
  plt.show()

####################################################################################################
## MAIN ############################################################################################
####################################################################################################

question_dict = {'question1': 1, 'question2': 2, 'question3': 3, 'question4': 4, 'question5': 5, 'question6': 6,
                 'question7': 7, 'question8': 8, 'question9': 9, 'question10': 10, 'question11': 11, 'question12': 12}

print('\n#################')
print('## Poject JEMA ##')
print('#################\n')

print('This is team JEMA\'s final project for CMSC 206.')

while True:
  selection = 0
  n = 0
  menu()

  while selection < 1 or selection > 13:

    if n == 0:
      selection = input('Please select from the menu above to see the results of each investigation or enter 13 to exit: ')
    else:
      selection = input('You must make a valid selection [1-12, 13 to exit]: ')

    try:
      selection = int(selection)
    except:
      selection = 0

    n += 1

  if selection == 13:
    print('\nProgram exiting...\n')
    break
  else:
    question_selector = 'question' + str(selection)
    for key in question_dict:
      if key == question_selector:
        eval(key+ '()')
