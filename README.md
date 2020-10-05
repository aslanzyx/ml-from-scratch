# ml-from-scratch
- Personal machine learning tools implementations - for learning usage only
- The majority of algorithms are really from course materials, but refactored in C# and with some additional features  
- The Note folder contains course notes

## Why C#
- All the class materials are originally writen in python. Python is a great language for scripting, but it is brainhurting when it comes to refactor code and optimize the algorithms because of its ducky typing. 
- C# is a rather lower-level language for data processing and machine learning (without ML.NET), however it provides some very robust basic data scructures like multi-dimensional array, which could help a lot to implement a data processing tool from scratch.
- This preoject is more about refactoring code from course materials and building a series of ML tools from scratch, so C# could be a very suitable language.
- Also Q# integration comes pretty handy when it comes to quantum ML later on.

## Classification

### Naive Bayes
- Take no param to construct
- The model stores 1 martix prob, 1 vector cond and 1 vector 
- Use **Fit** to train the model
- Use **Predict** to make predictions

### KNN
- Take an integer $k$ to construct
- Use **Fit** to train the model
  - No fitting process
  - The model is stroring all the data examples
- Use **Predict** to make predictions

## Clustering

### KMeans
- Take a $k$ as # of clusters and a series of initial means to construct
- Use **Fit** to train the model
  - By default cluster all examples at first and then adjust the means accordingly 
  - TODO: adjust means every some custome epochs
- Use **Predict**  to make prediction
  - By default uses Euclid distance

### TODO: 
- A dataset object in development but not yet integrated into models
