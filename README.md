# CS-5600-Porject

rosterfetch.py is used to fetch game day rosters and match results

combine.py is used to combine the seasons stats with the data gathered from rosterfetch.py

flatten.py takes the data from combine.py and flattens it into a usable vector

makesubs.py takes the flattened vecotrs and splits it into test, training, and validation data.

main.py is where the bulldozing happens.

To train a model simply run main.py. If the model exists - grader may need to change file paths for the where the model and data live - then main.py simply continues to train 
an old model. If the model has yet to be created main use model_factory to create a new one. To get the code running simply do pip installs of the needed packages. You should
only need pandas, tensorflow, pickle, numpy, sportsipy, and datetime to run the whole project. I may be worng about this but if I am missing one a simple pipisntall should fix the issues.

-Spencer Clemens
