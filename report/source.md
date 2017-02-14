% Starcraft 2 League Prediction
% Borja Lorente Escobar
% February 2017

Domain and Dataset
========
Starcraft 2 is a Real Time Strategy game, launched in 2010 by Blizzard. Since then, it has become one of the most popular competitive games of it's time, kickstarting the ascent of the E-sports into the mainstream.

As it is common in gaming, the community has gathered a collective knowledge of the characteristics of a good player. Some of these features are measurable, and players seeking to improve are often met with the advice that they analyze their scores in these aspects, and try to improve them. The most prominent measures of players skill are commonly:

* **Actions Per Minute** or **APM**, the number of mouse clicks, moves or key presses a player performs. The best players have an APM of around 400, while an average player will rarely go over 120.

Input Details
-------------
Data comes in csv files, accompanied with a `R` script that can be used to
merge data from both subjects, unfortunately when exporting that data structure
from `R` to `Octave` it is a collection of hashes, not a regular matrix, which
makes it harder to work with it.

Because of that and because the number of students from portuguese the subject
(649) was much greater than the number of math (395) we only included those
results from the portuguese subjects here, although in measures with the math
data set results were very close (around ±3% variance).

Because the csv file used characters to represent data we had to convert
everything to numbers, that procedure can be seen in `getData`.
```octave
function [X,y] = getData(filepath)
	preProcessData(filepath);

	data = csvread([filepath '.oct']);

	X = data(:, 1:32);
	y = data(:, 33);
end

function preProcessData(filepath)
	filestr = fileread(filepath);
	filestr = substr(filestr, 229);
	filestr = strrep(filestr, 'GP', '0');
	filestr = strrep(filestr, 'MS', '1');
	filestr = strrep(filestr, 'M', '0');
	filestr = strrep(filestr, 'F', '1');
	[...]
	filestr = strrep(filestr, '"', '');
	octcsv = fopen([filepath '.oct'], 'w');
	fprintf(octcsv, '%s', filestr);
end
```

Columns 1 through 30 are related to socio-economical status like parents
situation, alcohol consumption, whether they are on a relationship, etc...,
31 and 32 are first and second semester marks.

The Conundrum
-------------
As the original researchers did on their paper the attributes of the first and
second semester have a lot more weight than the other attributes.

As such here we included the final hit percentage of both options, with and
without previous years mark, but the graphics section refer only to the with
executions, as they were the ones that worked better and had more relevance.

On the original paper Business Intelligence methods worked the best in these
cases, but as they were out of the scope of this subject they were left out.

Methods Used
============

Classification Techniques
-------------------------
* Logistic Regression
* Neural Network
* Support Vector Machines

Testing Techniques
------------------
* Single Holdout Validation

Training Ratios
---------------
* 70 % Training Examples
* 10 % Adjustment Examples
* 20 % Validation Examples

Hyper-parameters Adjustment
---------------------------

Where two or more hyper-parameters where listed we created a matrix with all
possible combinations and iterated all over it.

### Logistic Regression
```octave
λ = [0,0.1,0.2,0.5,1,2,5,10,15,20,25];
```

Using `fminunc` with a max of 1000 iterations.

### Neural Networks
```octave
λ = [0,1,10,100];
hidden_nodes = [0,1,10,100];
```

Using `fmingc` with up to 1000 iterations were allowed, but the best result
hover over ~96 iterations.

### Support Vector Machines
```octave
C = [0.01,1,10,100];
σ = [0.01,1,10,100];
```

A Gaussian kernel was used, as it was the one which had the best results.

Results
=======

Percentages
-----------
This table reflects the best results achieved, changing not only,
hyper-parameters but also the confidence threshold, these can be
seen on both figures 1 and 5.

In bold the best percentage of each category.

| Method        | With previous marks | W/o previous marks |
|---------------|:-------------------:|:------------------:|
| LR            |        85.28%       |      **69.94%**    |
| NN            |      **88.72%**     |        69.74%      |
| SVM           |        83.97%       |        66.41%      |


Graphics
========

Following some graphs showing the breakdown of accuracy, recall, learning
curves or the adjustment planes of the different methods

![Logistic Regression Accuracy](graphs/logisticRegression/accuracy.png)

Logistic regression Accuracy is more or less stable even when changing the
threshold.

![Logistic Regression Learning](graphs/logisticRegression/learning.png)

![Logistic Regression Recall](graphs/logisticRegression/recall.png)

![Logistic Regression Adjustment](graphs/logisticRegression/adjustment.png)

![Neural Network Accuracy](graphs/neuralNetwork/accuracy.png)

![Neural Network Learning](graphs/neuralNetwork/learning.png)

![Neural Network Recall](graphs/neuralNetwork/recall.png)

![Support Vector Machine Adjustment](graphs/supportVectorMachine/adjusting.png)


Possible Improvements
=====================

K-Fold Cross Validation
-------------------------
`Matlab` `crossvalind` functions makes this pretty easy, unfortunately `Octave`
has no direct equivalent, and although it is not exceedingly hard to code it,
it is not trivial to make it generic enough so it works for every size of data.

There is a half-coded implementation of a 10-fold validation but it tended to
produce some errors with certain example sizes.

Parallelization
---------------
While the Neural Network in the code has parallel capacities because `fmincg`
(the gradient descent algorithm we use) seems smart enough to take advantage of
multiple cores the SVM and the LR did not had any parallel capacity, this meant
that on our 4-core machine we were wasting a lot of cycles.

To illustrate this even with this small set of data running the three strategies
could take upwards of half an hour, this is including the adjusting option on
all three, even then, seems like very poor performance with no scalability.

This could be trivialy parallelized with `Matlab` `parfor` syntax but in `Octave`
this is syntactic sugar for just a `for` loop, the only analogue option is the
`parallel` package in the Octave forge, unfortunately there wasn't just enough
time.

Better Stratification
---------------------
The stratification used this time was a bit *manual* in the sense that in order
to mix the slices a variable had to be be changed on the source code, it would
have been great to have automatic random stratification, but unfortunately,
again, not enough time.

Decision Trees
--------------
On the original paper the authors get a much better result with decision
trees and naive Bayes tha with plain machine learning algorithms, but because
we set out to not use any library but what was coded by us we didn't had any
implementation of decision trees at the time

Conclusions
===========
Predicting Student Performance only with socio-economical context is hard,
however with previous academic results we can not only provide an accurate
binary assertion on whether the student will pass or fail, but we can with
relative precision the range where the mark will be.

These results seems to indicate that with enough data machine learning could be
successfully applied to education analysis.

Another point to make is that even-though Machine Learning Techniques are not
the most adequate to try to examine and extract what attributes where taken
into account the most we can examine the theta vector given by the logistic
regression to infer the following:

* Trying to analyze & predict anything to do with human behavior is hard
* When we don't take into account the previous mark the absences and the
	desire to study higher education are the most important factors
	+ Age plays quite an important factor (The lower, the better)
	+ Doing extra activities is as important as going to paid extra classes
	+ Previous school years failures are also important (And related to age)

References
==========

[P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.](http://www3.dsi.uminho.pt/pcortez/student.pdf)

[Code & Data](https://github.com/AlvarBer/Students-Performance-Analysis)
