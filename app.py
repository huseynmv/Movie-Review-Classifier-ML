# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import f1_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# data = pd.read_csv('IMDB.csv')

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)

# # Convert the text data into feature vectors
# vectorizer = CountVectorizer(stop_words='english')
# X_train = vectorizer.fit_transform(X_train)
# X_test = vectorizer.transform(X_test)

# # Train the model
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# # Evaluate the model
# y_pred = clf.predict(X_test)
# f1 = f1_score(y_test, y_pred)
# print("F1 score: ", f1)

# # Generate a confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='g')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()


# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB

# # Read the CSV file
# data = pd.read_csv('IMDB.csv')
# reviews = data['review'].values
# sentiments = data['sentiment'].values

# # Convert text to numerical feature vectors using CountVectorizer
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(reviews)

# # Split the data into training and testing sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=42)

# # Train the model using Multinomial Naive Bayes
# clf = MultinomialNB()
# clf.fit(X_train, y_train)

# # Test the model on new input
# input_text = "This movie is great!"
# input_vector = vectorizer.transform([input_text])
# prediction = clf.predict(input_vector)

# print("Prediction:", prediction[0])

# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import classification_report

# # Load the data from the CSV file
# data = pd.read_csv('IMDB.csv')

# # Split the data into training and testing sets
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# # Convert the text data into a matrix of token counts
# vectorizer = CountVectorizer(stop_words='english')
# X_train = vectorizer.fit_transform(train_data['review'])
# X_test = vectorizer.transform(test_data['review'])

# # Prepare the target labels
# y_train = train_data['sentiment']
# y_test = test_data['sentiment']

# # Train a Naive Bayes model on the training data
# clf = MultinomialNB()
# clf.fit(X_train, y_train)

# # Evaluate the model on the testing data
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load the dataset using pandas
df = pd.read_csv('IMDB.csv', header=None, names=['review', 'sentiment'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Convert text data to numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Evaluate the model performance on the testing set
y_pred = model.predict(X_test_vectorized)
# print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix using matplotlib
# cm = confusion_matrix(y_test, y_pred)
# plt.matshow(cm)
# plt.title('Confusion matrix')
# plt.colorbar()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()
#my graphs
from sklearn.metrics import precision_recall_curve

# Compute precision and recall for each class
# precision, recall, _ = precision_recall_curve(y_test, y_pred, pos_label='positive')

# # Plot precision-recall curve
# plt.plot(recall, precision, label='Precision-Recall curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc='lower left')
# plt.show()
#my graphs

# Test the model on new input
# new_input = ["This movie was terrible! I would not recommend it to anyone."]
# new_input_vectorized = vectorizer.transform(new_input)
# print(model.predict(new_input_vectorized))

# import matplotlib.pyplot as plt
# import pandas as pd

# # load dataset
# data = pd.read_csv('IMDB.csv')

# # add a column for review length
# data['review'] = data['review'].apply(lambda x: len(x.split()))

# # create a scatter plot
# plt.scatter(data['review'], data['sentiment'])

# # add labels and title
# plt.xlabel('Review Length')
# plt.ylabel('Sentiment')
# plt.title('Correlation between Review and Sentiment')

# # display the plot
# plt.show()


# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# # Read in text from a file
# with open('IMDB.csv', 'r') as file:
#     text = file.read()

# # Create word cloud
# wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(text)

# # Display the generated image:
# plt.figure(figsize=(8, 8), facecolor=None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad=0)

# # Save the image
# wordcloud.to_file("wordcloud.png")



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
