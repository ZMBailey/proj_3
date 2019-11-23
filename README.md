# Understanding
A lot depends on the genre of a music and categorizing music into genres takes time. It takes human listening time, and while it is sometimes enojoyable, human listening may not be practical for a large number of songs. It may be possible to save human-hours by having a machine learning model analyze the songs for us! 

# Data
Approximately 100 songs were collected from 6 genres, Classical, Electronic, Jazz, Country, Metal and HipHop. Each song was broken up into 20 sec intervals to increase the size of the data set. These audio files were then processed using Librosa to extract sonic features that can help identify a songs genre. For an introduction into the music information retreival techniques, please see the related blog:
https://medium.com/@patrickbfuller/librosa-a-python-audio-libary-60014eeaccfb.

# Model
To classify the genres of the songs, a Support Vector Machine classifier was trained on vectors of sonic features present in the songs. Before training an 80 20 train test split was performed.

Based on sonic features, model predicts the probability that the set of features belongs to each of the 6 genres. A random guess would have a 17% chance of being right and our model predicted genre correctly 81% of the time.

To analyze a song from a youtube a custom class 'listens' to the song by extracting the features from the song using librosa and then passes those features into the trained model.

