import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("./bbc_data.csv")
df.columns = ['Text news', 'Category of the news']



# Transform category columns into numeric values
le = LabelEncoder()
df['Encoded category of the news'] = le.fit_transform(df['Category of the news'])


X_train, X_test, y_train, y_test = train_test_split(df['Text news'], df['Encoded category of the news'], test_size=0.2)
cv = CountVectorizer()
X_train_count = cv.fit_transform(X_train).toarray()
X_test_count = cv.transform(X_test).toarray()


model = MultinomialNB()
model.fit(X_train_count, y_train)
model.score(X_test_count, y_test)

text = ['Elon Musks accused OpenAI of turning the company into a profit rather than open source', 'Google launched a new AI model yesterday to take on ChatGPT']
prediction = model.predict(cv.transform(text))
le.inverse_transform(prediction)

