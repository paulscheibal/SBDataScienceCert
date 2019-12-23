This document describes the folders and the contents of the directories:

The folders are as follows:

Data - contains the main data csv files for the programs

DataWrangling - folder which contains the data wrangling programs

Validation - folder which contains the validation program validating wrangled Lahman data against FanGraphs

Discovery - folder which contains the programs for EDA.

Statistics - folder which contains the programs for statistical analysis.

MachineLearning - folder which contains the programs used for results of machine learning algorthms against the machine learning model

ReportOuts - folder containing documents for report outs including final document.


The files in the folders are as follows:

Data
	dfbatting_player_stats.csv - is a CSV file which was the output of the BaseballDataWrangling.py program.  It is the fundamental data
		used throughout the project.  Data is batting performance from 1954 to 2018 of the Sean Lahman data.
	dfbatting_player_stats_OPS.csv - is dfbatting_player_stats data with only players who had at least 300 AB for a given year.  Lag1 one statistics
		were added to this CSV file as features for the predictive model.  OPS and career OPS were predicted.
	dfbatting_player_stats_OPS_all.csv - is dfbatting_player_stats data with with lag1 one statistics added to this CSV file as features 
		for the predictive model.  OPS and career OPS were predicted.
	dfrtm_OPS.csv - CSV file with rtm data using the binomial estimator.

DataWrangling
	BaseballDataWrangling.py - this program inputs the Sean Lahman baseball yearly statistics from 1871 to 2018 and outputs dfbatting_player_stats
		which is a wrangled version of batting performance from 1954 to 2018 to be used for machine learning.
	BaseballDataWranglingRTMbyyear.py - program which calculates the regression towards the mean estimators using the binomial estimator.
	BaseballDataWraninglinglagcalculations.py - program which calculates the lag1 values for input to the predictive model.

Validation 
	BaseballDataValidation.py - inputs the dfbatting_player_stats and validates it against Fangraphs batting performance data.

Discovery
	BaseballExploratoryDataAnalysisandDiscovery.py - program which performs EDA plots.  
	DataStory.ipynb - Jupyter Notebook which shows the EDA plots in a readable fashion.

Statistics
	BaseballStatistics.py - program which performs a series of statisical analysis.
	StaticalAnalyisStory.ipynb - Jupyter Notebook which shows the statisical analyis in a readable fashion.

MachineLearning
	BaseballMachineLearning_OPS.py - program which runs a series of algorithms against the baseball performance predictive model using OPS as the 
		predicted metric.
	BaseballMachineLearning_careerOPS.py - program which runs a series of algorithms against the baseball performance predictive model using career OPS as the 
		predicted metric.
	BaseballMachineLearningSelectedPlayerPredictions.py - program which runs a selected set of players against a set of machine learning algorithms.
	BaseballMLClassificationPOS.py - program which classifies a player as either a shortstop or 1st baseman based upon feature input to the model.
	BaseballFiveYearProjections.py - program which creates a file of players with five year forecasts to be used as input to the ML algorithms.
	BaseballFiveYearProjecttionsPlot.py - program which simply plots the best R Square results of 5 year predictive model.
	BaseballConsistentvsNonConsistentPlayers.py - program which takes as input all players playing 6 years and outputs the top 20 players with the 
		highest OPS variance and the top 20 players with the lowest variance of OPS. 
	MachineLearningStory.ipynb - Jupiter Notebook which shows the machine learning algorithms results using OPS as the predictive metric.
	MachineLearningStoryCareerOPS.ipynb - Jupiter Notebook which shows the machine learning algorithms results using career OPS as the predictive metric.
	

