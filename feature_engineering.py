import pandas as pd
import numpy as np
from datetime import datetime



def convert_date(df):
	df["Year"],df["Month"],df["WeekDay"] = zip(*df["DateTime"].map(get_year_month_week))
	df.drop(["DateTime"],axis=1,inplace=True)
	

def get_year_month_week(dt):
	d = datetime.strptime(dt,"%Y-%m-%d %H:%M:%S")
	return d.year, d.month, d.isoweekday()

def age_to_months(age):
	if age is np.nan:
		return 12.0
	v1,v2 = age.split()
	if v2 in ["year",'years']:
		return float(v1) * 12
	elif v2 in ['month','months']:
		return float(v1) 
	elif v2 in ['week','weeks']:
		return float(v1) / 4.5
	elif v2 in ['day','days']:
		return float(v1) / 31


def get_dummy_data_y(df):
	df = pd.get_dummies(df)
	return df


def get_dummy_data(train_data,test_data):
	categorical_variables = ['AnimalType', 'SexuponOutcome', 'Breed','Year', 'Color', 'Month',\
	'WeekDay','Named','TypeOfColor','AgeCategory']
	# categorical_variables = ['AnimalType', 'SexuponOutcome', 'Breed', 'Color','WeekDay']
	train_data["Train"] = 1
	test_data["Train"] = 0

	concat = pd.concat([train_data,test_data])
	concat_encoded = pd.get_dummies(concat,columns=categorical_variables)

	train = concat_encoded[concat_encoded["Train"] == 1]
	test = concat_encoded[concat_encoded["Train"] == 0]
	train.drop(["Train"],axis=1,inplace=True)
	test.drop(["Train"],axis=1,inplace=True)

	return train,test


def convert_age_to_months(df):
	df["AgeuponOutcome"] = df["AgeuponOutcome"].map(age_to_months)
	# df['AgeuponOutcome'] = (df['AgeuponOutcome'] - df['AgeuponOutcome'].min())/(df['AgeuponOutcome'].max() - df['AgeuponOutcome'].min())
	# print df["AgeuponOutcome"].median()
	# exit(0)
	
def drop_columns(train_data,test_data):
	mapping = {'Adoption':'0','Died':'1','Euthanasia':'2','Return_to_owner':'3','Transfer':'4'}
	train_data.replace({'OutcomeType':mapping},inplace=True)

	train_id = train_data[["AnimalID"]]
	test_id = test_data[["ID"]]
	train_outcome = train_data["OutcomeType"]

	############ check animal named or not ################
	train_data['Named'] = np.where(train_data['Name'].notnull(),'yes','no')
	test_data['Named'] = np.where(test_data['Name'].notnull(),'yes','no')


	########### check mixed or pure breed ##########
	# check_mixed = '|'.join(['/','Mix'])
	# train_data['TypeOfBreed'] = np.where(train_data.Breed.str.contains(check_mixed),"mixed","pure")
	# test_data['TypeOfBreed'] = np.where(test_data.Breed.str.contains(check_mixed),"mixed","pure")
	

	########### Single or mixed color ###############
	train_data['TypeOfColor'] = np.where(train_data.Color.str.contains("/"),"multi color","single color")
	test_data['TypeOfColor'] = np.where(test_data.Color.str.contains("/"),"multi color","single color")
	# train_data.drop(["Color"],axis=1,inplace=True)
	# test_data.drop(["Color"],axis=1,inplace=True)	

	########### Pup, young, adult or senior dog or cat #########
	train_data.loc[(train_data['AgeuponOutcome'] <= 7) & (train_data['AnimalType'] == 'Dog'),'AgeCategory'] = '1'
	train_data.loc[(train_data['AgeuponOutcome'] > 7) & (train_data['AgeuponOutcome'] <= 24) & (train_data['AnimalType'] == 'Dog'),'AgeCategory'] = '2'
	train_data.loc[(train_data['AgeuponOutcome'] > 24) & (train_data['AgeuponOutcome'] <= 96) & (train_data['AnimalType'] == 'Dog'),'AgeCategory'] = '3'
	train_data.loc[(train_data['AgeuponOutcome'] > 96) & (train_data['AnimalType'] == 'Dog'),'AgeCategory'] = '4'
	
	train_data.loc[(train_data['AgeuponOutcome'] <= 8) & (train_data['AnimalType'] == 'Cat'),'AgeCategory'] = '1'
	train_data.loc[(train_data['AgeuponOutcome'] > 8) & (train_data['AgeuponOutcome'] <= 24) & (train_data['AnimalType'] == 'Cat'),'AgeCategory'] = '2'
	train_data.loc[(train_data['AgeuponOutcome'] > 24) & (train_data['AgeuponOutcome'] <= 96) & (train_data['AnimalType'] == 'Cat'),'AgeCategory'] = '3'
	train_data.loc[(train_data['AgeuponOutcome'] > 96) & (train_data['AnimalType'] == 'Cat'),'AgeCategory'] = '4'
	
	test_data.loc[(test_data['AgeuponOutcome'] <= 7) & (test_data['AnimalType'] == 'Dog'),'AgeCategory'] = '1'
	test_data.loc[(test_data['AgeuponOutcome'] > 7) & (test_data['AgeuponOutcome'] <= 24) & (test_data['AnimalType'] == 'Dog'),'AgeCategory'] = '2'
	test_data.loc[(test_data['AgeuponOutcome'] > 24) & (test_data['AgeuponOutcome'] <= 96) & (test_data['AnimalType'] == 'Dog'),'AgeCategory'] = '3'
	test_data.loc[(test_data['AgeuponOutcome'] > 96) & (test_data['AnimalType'] == 'Dog'),'AgeCategory'] = '4'
	
	test_data.loc[(test_data['AgeuponOutcome'] <= 8) & (test_data['AnimalType'] == 'Cat'),'AgeCategory'] = '1'
	test_data.loc[(test_data['AgeuponOutcome'] > 8) & (test_data['AgeuponOutcome'] <= 24) & (test_data['AnimalType'] == 'Cat'),'AgeCategory'] = '2'
	test_data.loc[(test_data['AgeuponOutcome'] > 24) & (test_data['AgeuponOutcome'] <= 96) & (test_data['AnimalType'] == 'Cat'),'AgeCategory'] = '3'
	test_data.loc[(test_data['AgeuponOutcome'] > 96) & (test_data['AnimalType'] == 'Cat'),'AgeCategory'] = '4'
	

	train_data.drop(["AnimalID","Name","OutcomeSubtype","OutcomeType","AgeuponOutcome"],axis=1,inplace=True)
	test_data.drop(["ID","Name","AgeuponOutcome"],axis=1,inplace=True)
	
	return train_id,test_id,train_outcome


