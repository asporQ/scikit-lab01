from sklearn import tree

features = [[120, 5], [110, 5], [180, 12], [163, 12], [93, 5]]
labels = ["ECO CAR / B-SEGMENT", "ECO CAR / B-SEGMENT", "VAN", "VAN", "NORMAL"]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)
print(classifier.predict([[200, 5]]))  # it will print predicted car
