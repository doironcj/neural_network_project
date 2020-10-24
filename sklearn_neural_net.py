from sklearn.neural_network import MLPClassifier




#testing data sets
file_test_data = open("a2-test-data.txt","r")
file_test_label = open("a2-test-label.txt","r")
test_data  = []
test_label = []
for line in file_test_data:
    words = line.split()
    set = []
    for num in words:
        set.append(float(num))
    test_data.append(set.copy())

for label in file_test_label:
    test_label.append(float(label))

# trainig data sets 
file_data = open("a2-train-data.txt","r")
file_label = open("a2-train-label.txt","r")
training_data  = []
training_label = []
for line in file_data:
    words = line.split()
    set = []
    for num in words:
        set.append(float(num))
    training_data.append(set.copy())

for label in file_label:
    training_label.append(float(label))
classifier = MLPClassifier(solver="adam", alpha = .00001, hidden_layer_sizes=(25), random_state=1)
classifier.fit(training_data,training_label)

p = classifier.predict(training_data)

num_correct = 0

for i in range(0,len(p)):
    if (p[i] == training_label[i]):
        num_correct += 1

print(num_correct)
num_correct = 0
test_p = classifier.predict(test_data)
for i in range(0,len(test_p)):
    if (test_p[i] == test_label[i]):
        num_correct += 1
print("percent accuracy for test data:")
print(num_correct/len(test_p))
print(test_p)
#print(p)

