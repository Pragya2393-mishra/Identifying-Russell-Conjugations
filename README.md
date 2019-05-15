# Identifying-Russell-Conjugations

#How to run the file:

1. Download pre-trained google wordvector files (both files) from this link: https://drive.google.com/drive/folders/1f0mQ2cgcx9EWVzJjwSw_3X-7w_NKshTj?usp=sharing
2. Final code for approach 1 is present in the path: Project Execution\Models\Approach1\Pipeline\Final version\Final_pipeline_approach1.py
3. Run this file, after changing the path of the google normed vectors to the path where the files were saved to in step(1)
4. Path to datasets used for training the classifiers:
  data1: Project Execution\Data\Approach1\Final versions\data_contexual_denotational_grouping.csv
  data2: Project Execution\Data\Approach1\Final versions\data_sentiment_classification.csv
5. Path to the Validation dataset:
  test1: Project Execution\Test files\Approach1\Final test file\Validation set.csv
  
#Project report and summary can be found in the folder : \Final report
 
#Introduction to the project:
Objective: To build and deploy a prototype bias-revealing browser plugin, which will reveal hidden sources of emotive bias (Russell Conjugates) in online rhetoric. The long-term goal of this project is to investigate the extent to which Russell conjugations are used to bias rhetoric, to develop tools to make readers aware of such rhetorical tricks, and to investigate how such tools affect readers’ perceptions of bias and evaluation of information. The scope of the current project is to build a browser plug-in that automatically
a)	Identifies source of polarization in a rhetoric, i.e., emotively connotated words,
b)	identify and present Russell Conjugates of these emotive words as an option to the user.

Background: The use of subtly emotive language can bias interpretation of otherwise objective and accurate characterizations of people and events. Speechwriters and rhetoricians have used careful word choice to good effect since time immemorial. Bertrand Russell memorably encapsulated the idea in pseudo-conjugations such as:
•	I am firm, you are obstinate, he is a pig-headed fool.
•	I am righteously indignant, you are annoyed, he is making a fuss over nothing.
•	I have reconsidered the matter, you have changed your mind, he has gone back on his word.
Here, pairs of words such as ‘firm’, ‘obstinate’ and ‘pigheaded’ are known as Russell or Emotive Conjugates of each other.  As rhetoric in all media, both political and non-, has become increasingly polarized, so has, it seems, the use of such emotive language to pre-emptively destroy one’s opponents and prop-up one’s heroes.



