% Starcraft II League Prediction
% Borja Lorente Escobar
% February 2017

Domain and Motivation
========
Starcraft II is a Real Time Strategy game, launched in 2010 by Blizzard. Since then, it has become one of the most popular competitive games of it's time, kickstarting the ascent of the E-sports into the mainstream.

As it is common in gaming, the community has gathered a collective knowledge of the characteristics of a good player. Some of these features are measurable, and players seeking to improve are often met with the advice that they analyze their scores in these aspects, and try to improve them. The most prominent measures of players skill are commonly:

* **Actions Per Minute** or **APM**, the number of mouse clicks, moves or key presses a player performs. The best players have an APM of around 400, while an average player will rarely go over 120.
* **Workers Created**. In Starcraft II, the player with the most resources usually wins the game, and thus it is deemed important for a good player to constantly be creating workers to gather as many resources as possible.
* **Minimap Clicks**. Since a Starcraft II player has to control every single unit in the team, many places often require immediate attention, such as jumping from a battle on one side of the map to creating new workers back at the base. This is why efficient camera movement is vital for a player, and therefore a good player will often click on the minimap to move the camera, instead of dragging it to the sides, as it is much faster.

The aim of this study is to determine how useful these player traits are when trying to predict the skill level of a player. In the case that they're not very significant, a secondary goal is to determine _which_ attributes are truly determinant of a player's skill.

To that end, the analyzed data has been extracted from game replays of real users of every ranking.

Dataset Details
-------------
The [extracted data](1) comes in the form of a `.csv` file containing data from over 3300 games, one data point per line, organized by columns denoting the various values of the studied attributes:

|       |         |           |
|------------------|--------------------|-----------------|
| GameID           | **LeagueIndex**        | Age             |
| HoursPerWeek     | TotalHours         | **APM**           |
| SelectByHotkeys  | AssignToHotkeys    | UniqueHotkeys   |
| MinimapAttacks   | **MinimapRightClicks** | NumberOfPACs    |
| GapBetweenPACs   | ActionLatency      | ActionsInPAC    |
| TotalMapExplored | **WorkersMade**        | UniqueUnitsMade |
| ComplexUnitsMade | ComplexAbilityUsed | MaxTimeStamp    |

The **LeagueIndex** attribute denotes the rank of the player, from Bronze to Diamod, with a higher number indicating a higher rating.

To preprocess the data, `Octave`'s built-in `.csv` manipulation tools were used, by loading the file and stripping the header:

```octave
raw_data = dlmread('rawdata/starcraft.csv', ',');
% Ignore the first row (names)
raw_data = raw_data(2:rows(raw_data), :);
% Select base attributes
[X, y] = selectBaseAttributes(raw_data);
% Split samples into training, validation and test sets
[X_train, y_train, X_val, y_val, X_test, y_test] = splitSamples(X, y);
```

This structure gives the necessary flexibility to study the effects of different attribute selections (by modifying `selectBaseAttributes`), and to create arbitrary sets of samples for test and validation.

After some adjustments, the following split was fould the most satisfying for the distribution of samples, which were obtained by a random shuffling of the samples with the following code:

| Training | Validation | Test |
|----------|------------|------|
| 80%      | 10%        | 10%  |

```octave
function [X_train, y_train, X_val, y_val, X_test, y_test] = splitSamples(X, y)
	m = rows(X);

	train_percent = 0.8;
	val_percent = 0.1;
	test_percent = 0.1;
	train_size = floor(train_percent * m);
	val_size = floor(val_percent * m);
	test_size = floor(test_percent * m);

	index_vector = randperm(m);
	index = 1;
	train_indices = index_vector(index:(index + train_size));
	index += train_size;
	val_indices = index_vector(index:(index + val_size));
	index += val_size;
	test_indices = index_vector(index:(index + test_size));

	X_train = X(train_indices, :);
	y_train = y(train_indices, :);
	X_val = X(val_indices, :);
	y_val = y(val_indices, :);
	X_test = X(test_indices, :);
	y_test = y(test_indices, :);
end  % splitSamples
```

Methodology
============

As this was an exploratory study, there were several iterations over the classification techniques used.

However, some things became apparent from the beginning, such as the inability of **Support Vector Machines** to correctly predict the league of a player, **never reaching an accuracy of 10% over the set samples**. Therefore, even though the used hyper-parameters will be explained in the following section and the source code can be found in the annex, the results have been omitted from the discussion of the different iterations. The focus of the results is therefore on the **Logistic Regression** and **Neural Networks**.

Hyper-parameters Adjustment
---------------------------

Where two or more hyper-parameters where listed the program iterated over all the possible combinations of the sets.

### Logistic Regression
```octave
λ = 0:0.01:0.5;
```

Using `fminunc` with a max of 5000 iterations.

### Neural Networks
```octave
λ = 0:0.1:5;
hidden_layer_sizes = [5, 10, 25, 50, 100, 150];
```

Using `fmingc` with up to 5000 iterations were allowed. With hidden layer sizes above 25, all iterations were used up by the function. In addition to that, higher layer sizes seem to predict better with higher `λ`s.

In the result graphs, only the best graph is shown out of all the hidden layer sizes.

### Support Vector Machines
```octave
CSigma_seeds = 0.01:0.01:0.04;
CSigma_iterations = 5;
```
Here the C and sigma values were created by iterating over the `CSigma_seeds` vector in the following manner:

```octave
function [pool] = generateValuePool(seeds, iterations)
	pool = zeros(columns(seeds), iterations);
	for i = 0:iterations
		pool(:, i + 1) = seeds .* (10 ^ i);
	endfor
	% Vectorize pool
	poolsize = numel(pool);
	pool = reshape(pool, 1, poolsize);
endfunction

```

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
