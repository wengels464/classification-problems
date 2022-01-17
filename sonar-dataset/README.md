# Sonar Dataset SVM Example

## Changelog

I modified the original CSV located at 

http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)

to include a "targets" header in the last column and numerical ascending headers for the other columns.

These changes are present now in the data but were made outside of Python.

## Method

I decided upon a SVM because I was convinced that due to the geometrically-dependent nature of the sonar readings the data would be linearly separable.

A polynomial kernel was decided upon by manual experimentation, but I am still unsure why it outperforms RBF and Linear.

A standard test size of 20% was used.

Hyperparameter tuning for **degree** and **coeff0** was carried out manually using *accuracy_score* from sklearn as an indicator.

## Tools

Data from the original CSV was read via Pandas

The Sklearn Suite was used heavily.

SVC was the SVM algorithm of choice owing to the option to choose different kernels.
