When we talk about XGBoost and RandomForest there is no better model as both of them serve their own purpose and both have their set of advantages or disadvantages.

While the common assumption is that XGBoost is proven to be better than Random Forest as it is a sequential model which keeps iteratively training itself, ideally that's not always the case.
So let's get a brief idea about both the models' advantages and drawbacks. (PS: Advantages and Drawbacks are in reference to the models, and not necessarily true in all scenarios, and not all cases covered only the ones relevant to a comparison between the two)

RandomForest is basically a series of independent decision trees (as the name suggest a forest made of trees) which combines their results to make the final decision or prediction. It might vote out or average out the results.

Advantages of Random Forest:-

 All the trees are trained on a randomly sampled subset of the training data, this randomization helps in reducing overfitting and also helps in dealing with outliers and missing data.
 Another advantage of Random Forest is, it can handle high-dimensional datasets as it is computationally efficient compared to XGBoost.
 It can deal with complicated datasets where there is a complex interaction between features.

Few Drawbacks of Random Forest:-

As it is a combination of a large number of decision trees it could take up a lot of resources in the form of memory and storage, especially for large datasets with numerous trees. This may at times limit the scalability of the model.

When the number of features outnumbers the number of samples in the data RandomForest wouldn't be the ideal choice compared to XGBoost as with more features it might struggle to find meaningful patterns or relationships between the features and the target variable. This is known as the curse of dimensionality.

With so many combinations of decision trees at times it becomes really difficult to interpret the model and find the exact relation between the features and the predicted results.

XGBoost is a boosting algorithm that sequentially builds an ensemble of decision trees with each subsequent tree correcting the mistakes made by the previous tree. XGBoost basically uses a gradient framework where each new tree learns from the residual of the previous trees.

Advantages of XGBoost:-

The iterative nature of XGBoost makes it well-suited for tasks where fine-tuning and high predictive performance are needed. XGBoost slowly and gradually improves the entire model by refining its predictions at each step. 

XGBoost is a highly optimized algorithm with great speed and efficiency due to its parallel processing and column block storage technique which makes it faster and memory efficient than Random Forest. Also as it can be distributed across several machines it can be integrated during big data processing.

It is a more flexible model, it allows you to set your own regularisation technique L1 or L2 and tree complexity as well (gamma). We can even define our own custom loss functions or objective functions as per the optimization required for our model.

Here unlike Random Forest, we do get the relation of features with the data and we can even rank the importance of input features in the model's predictions.

Few Drawbacks of XGBoost:-

It is computationally very expensive due to its sequential nature and the need to update weights after each tree.

It can take a lot of time to train XGBoost as it has several hyper-parameters which are needed to be tuned carefully to get ideal results and this could take a lot of time and experimentation.

As compared to Random Forest XGBoost is more prone to overfitting especially when the hyper-parameters are not properly tuned. If there is a complex model which is overfitted it can capture noise or irrelevant patterns in the data, leading to decreased generalization performance on unseen data(Test and Validation)

Here I experimented, with housing data to predict the sale price which I got from Kaggle ( https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data ) and even I had the general assumption that XGBoost will outperform RandomForest. But surprisingly I found almost similar results with the two models the mean absolute error and r2 scores are identical.I used grid search for both models to find the correct set of hyperparameters

So the selection of a model can depend on a lot of factors as an ideal practice it is good to try out both of the models on the dataset and then decide for yourself.
Feel free to drop any comments if you have or even if you feel I was wrong or missed something would be more than happy to learn.

