# How I Predicted Ames Housing Prices

When I first opened this dataset, I wanted to actually understand what I was working with before jumping into models. So I started by looking at how sale prices were distributed, and they turned out to be pretty skewed — a handful of really expensive homes pull the average up. That one observation ended up shaping most of what came after, since I decided to model on the log of price instead of raw dollars.

From there I dug into what actually drives price here. Overall quality and living area came out as the two strongest single predictors, which isn't exactly shocking, but neighborhood mattered more than I expected — median prices ranged from around $90k to over $300k depending on where a house sits. I also caught two homes that were huge and top-quality but sold for way less than they should have — classic outliers that I ended up dropping since they were distorting the fit.

Missing data took more care than I initially expected. A lot of the "missing" values weren't actually missing at all — no pool, no basement, no fireplace just gets recorded as blank in this dataset instead of an explicit "none." So I went column by column and filled those in properly based on what they actually meant, and handled the genuinely missing values separately — for instance, using neighborhood-level medians for missing lot sizes instead of one global number that wouldn't reflect local differences.

For features, I converted the quality ratings (Excellent, Good, Average, and so on) into an actual numeric scale so the model could use the ordering instead of treating them as unrelated categories. I also built a few combined features — total square footage, house age, total bathrooms — that ended up being more useful than the individual pieces they were built from.

On the modeling side, I compared several approaches side by side using 5-fold cross-validation: a simple baseline, Ridge regression, Lasso, Random Forest, and Gradient Boosting. Lasso and a tuned Elastic Net ended up beating the tree-based models pretty clearly — I think that comes down to all the feature engineering, which made the relationships more linear and played to the strengths of a linear model over something like Random Forest. My best single model was an Elastic Net.

In the end, I got a further boost by blending that Elastic Net with a Gradient Boosting model (65/35 weighting), since the two tend to get different houses wrong and averaging them smoothed out some of each model's individual mistakes.

That final blend, retrained on the full training set, is what generated the predictions in this submission — cross-validated RMSE of about 0.108 on the log scale.
