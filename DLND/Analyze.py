import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)

group_by_day = rides['cnt'].groupby(rides['dteday'])
group_by_holiday = rides['cnt'].groupby(rides['holiday'])
group_by_weekday = rides['cnt'].groupby(rides['weekday'])
group_by_workingday = rides['cnt'].groupby(rides['workingday'])

for date, bikescount in group_by_day:
	#break
	print('date:{0}, bikes:{1}'.format(date, bikescount.sum()))
	
for date, bikescount in group_by_weekday:
	break
	#print('date:{0}, bikes:{1}'.format(date, bikescount.sum()))
	
for date, bikescount in group_by_holiday:
	break
	#print('date:{0}, bikes:{1}'.format(date, bikescount.sum()))
	
for date, bikescount in group_by_workingday:
	break
	#print('date:{0}, bikes:{1}'.format(date, bikescount.sum()))