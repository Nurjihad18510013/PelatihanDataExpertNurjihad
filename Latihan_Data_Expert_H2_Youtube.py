#!/usr/bin/env python
# coding: utf-8

# In[2]:


from textwrap import wrap

import emoji
import joblib
import langdetect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")


# In[3]:


def get_category_dict(category_file):
    category = pd.read_json(category_file, orient="records")
    category = pd.DataFrame(category["items"].values.tolist())
    
    return {
        cat.id: cat.snippet.get("title")
        for cat in category.itertuples(index=False)
    }


# In[4]:


category_dict = get_category_dict("C:\Data_Expert\category.json")


# In[5]:


trending = pd.read_csv("C:/Data_Expert/trending.csv", parse_dates=["publish_time", "trending_time"])

with pd.option_context("display.max_columns", None):
    display(trending.head())


# In[6]:


start_date = trending.trending_time.min()
end_date = trending.trending_time.max()

print(f"{start_date = }")
print(f"{end_date = }")


# In[7]:


filtered_trending = trending[trending.trending_time.dt.month >= 7]

start_date = filtered_trending.trending_time.min()
end_date = filtered_trending.trending_time.max()

print(f"{start_date = }")
print(f"{end_date = }")


# In[8]:


num_videos = filtered_trending.shape[0]
print(f"{num_videos = }")


# In[9]:


filtered_trending.info()


# In[10]:


filtered_trending.dropna(subset=["description"], inplace=True)


# In[11]:


trending_by_date = filtered_trending.groupby(
filtered_trending.trending_time.dt.date
)
num_trending_per_day = trending_by_date.trending_time.count()
print("Number of videos in trending per day:", num_trending_per_day.unique())


# In[12]:


trending_duration = filtered_trending.groupby("title").agg(
trending_duration=pd.NamedAgg(column="trending_time", aggfunc="count"),
trending_start_date=pd.NamedAgg(column="trending_time", aggfunc="min"),
trending_last_date=pd.NamedAgg(column="trending_time", aggfunc="max")
).sort_values("trending_duration", ascending=False).reset_index()

trending_duration.head(10)


# In[13]:


plt.figure(figsize=(15, 6))
plt.bar(
    trending_duration.title[:10].apply(lambda title: "\n".join(wrap(title, width=10))),
    trending_duration.trending_duration[:10]
)
plt.title("Longest Duration of videos included in Youtube Trending Video", loc="left")
plt.xlabel("Videos Title")
plt.ylabel("Trending Duration (in days)")
plt.grid(False)
plt.show()


# In[14]:


trending_by_title = filtered_trending.groupby("title")
trending_rewind = trending_by_title[["view", "like", "dislike"]].agg(["min", "max", "mean", "sum"])
trending_rewind


# In[15]:


top_10_liked = trending_rewind["like"].sort_values("max", ascending=False).iloc[:10]

plt.figure(figsize=(15, 6))
plt.bar(
    top_10_liked.index.to_series().apply(lambda title: "\n".join(wrap(title, width=10))),
    top_10_liked["max"],
    label="last like"
)
plt.bar(
    top_10_liked.index.to_series().apply(lambda title: "\n".join(wrap(title, width=10))),
    top_10_liked["min"],
    label="start like"
)
plt.title("Most videos in trending list improves drestically in terms of likes", loc="left", y=1.1)
plt.xlabel("Number of Like")
plt.ylabel("Video Title")
plt.legend()
plt.grid(False)
plt.show()


# In[16]:


top_10_viewed = trending_rewind["view"].sort_values("max", ascending=False).iloc[:10]

plt.figure(figsize=(15, 6))
plt.bar(
    top_10_viewed.index.to_series().apply(lambda title: "\n".join(wrap(title, width=10))),
    top_10_viewed["max"],
    label="last views"
)
plt.bar(
    top_10_viewed.index.to_series().apply(lambda title: "\n".join(wrap(title, width=10))),
    top_10_viewed["min"],
    label="start views"
)
plt.title("Most videos in trending list improves drestically in terms of views", loc="left", y=1.1)
plt.xlabel("Number of Views")
plt.ylabel("Video Title")
plt.legend()
plt.grid(False)
plt.show()


# In[17]:


top_10_disliked = trending_rewind["like"].sort_values("max", ascending=False).iloc[:10]

plt.figure(figsize=(15, 6))
plt.bar(
    top_10_disliked.index.to_series().apply(lambda title: "\n".join(wrap(title, width=10))),
    top_10_disliked["max"],
    label="last dislike"
)
plt.bar(
    top_10_disliked.index.to_series().apply(lambda title: "\n".join(wrap(title, width=10))),
    top_10_disliked["min"],
    label="start dislike"
)
plt.title("Most videos in trending list improves drestically in terms of dislikes", loc="left", y=1.1)
plt.xlabel("Number of Dislike")
plt.ylabel("Video Title")
plt.legend()
plt.grid(False)
plt.show()


# In[18]:


sample = filtered_trending.sample(10, random_state=11)
sample[["title", "description"]]


# In[19]:


sample["title_lang"] = sample.title.apply(lambda title: langdetect.detect(title.lower()))
sample["desc_lang"] = sample.description.apply(lambda desc: langdetect.detect(desc.lower()))


# In[20]:


with pd.option_context("display.max_colwidth", 100):
    display(sample[["title", "title_lang", "description", "desc_lang"]])


# In[21]:


def detect_language(text):
    """Detect language of the 'text'."""
    try:
        lang = langdetect.detect(text)
        return lang
    except:
        return


# In[22]:


filtered_trending["title_lang"] = filtered_trending["title"].apply(detect_language)
filtered_trending["desc_lang"] = filtered_trending["description"].apply(detect_language)


# In[23]:


filtered_trending[["title", "title_lang", "description", "desc_lang"]]


# In[24]:


indo_trending = filtered_trending.loc[
    (filtered_trending.title_lang == "id") | (filtered_trending.desc_lang == "id")
]

with pd.option_context("display.max_columns", None):
    display(indo_trending.sample(10))


# In[50]:


data = indo_trending[["title", "description", "category_id"]].reset_index(drop=True)
data


# In[51]:


data.drop_duplicates(subset="title", inplace=True)


# In[52]:


data.reset_index(drop=True, inplace=True)
data.shape


# In[53]:


import emoji
list_emoji = [e for e in emoji.UNICODE_EMOJI.get("en")]
count = 0
for em in list_emoji:
    for title in data.title:
        if em in title:
            count += 1
print("How many titles use emoji?", count)


# In[54]:


def demojize(text):
    for em in list_emoji:
        if em in title:
            em_text = emoji.demojize(em)
            text = text.replace(en, " " + em_text + " ")
    return text


# In[55]:


data["title_emoji"] = data.title.apply(demojize)


# In[56]:


title_width_emoji_idx =[
    idx for idx in range(len(data.title))
    for em in list_emoji
    if em in data.loc[idx, "title"]
]


# In[41]:


with pd.option_context("display.max_colwidth", 100):
    display(data.loc[title_width_emoji_idx])


# In[57]:


data.drop(columns="title", inplace=True)


# In[58]:


desc_width_emoji_idx = [
 idx for idx in range(len(data.description))
 for em in list_emoji
 if em in data.loc[idx, "description"]
]
data["desc_emoji"] = data.description.apply(demojize)
with pd.option_context("display.max_colwidth", 100):
 display(data.loc[desc_width_emoji_idx])


# In[59]:


data.drop(columns="description", inplace=True)


# In[60]:


data["all_text"] = data["title_emoji"] + " " + data["desc_emoji"]


# In[69]:


X_train, X_dev, y_train, y_dev = train_test_split(
    data["all_text"], data["category_id"],
    test_size=.2,
    stratify=data["category_id"],
    random_state=11
)
training_size = X_train.shape[0]
dev_size = X_dev.shape[0]
print(f"{training_size = }.. {dev_size = }")

vectorizer = TfidfVectorizer(
    min_df=.15,
    max_df=.7,
    ngram_range=(1, 1),
)

train_tfidf = vectorizer.fit_transform(X_train)
dev_tfidf = vectorizer.transform(X_dev)
print("Got train tf-idf with shape:", train_tfidf.shape)
print("Got dev tf-idf with shape:", dev_tfidf.shape)

train_tfidf = pd.DataFrame(train_tfidf.toarray(), columns=vectorizer.get_feature_names())
dev_tfidf = pd.DataFrame(dev_tfidf.toarray(), columns=vectorizer.get_feature_names())


# In[70]:


with pd.option_context("display.max_columns", 100):
    display(train_tfidf.sample(5))


# In[71]:


dict_models = {
    "logistic_regression": LogisticRegression(),
    "naive_bayes": MultinomialNB(),
    "svm": LinearSVC(random_state=11),
    "decision_tree": DecisionTreeClassifier(random_state=11),
    "random_tree": RandomForestClassifier(random_state=11)
}


# In[72]:


for model in dict_models.values():
    print(f"-- {model.__class__.__name__} --")
    model.fit(train_tfidf, y_train)
    y_pred = model.predict(dev_tfidf)
    print("Reports on dev set:", classification_report(y_dev, y_pred), sep="\n")


# In[75]:


svm_grid_search = GridSearchCV(
    dict_models["svm"],
    {"C": (10, 1, .1, .05, .01)},
)

svm_grid_search.fit(train_tfidf, y_train)
svm_pred_dev = svm_grid_search.predict(dev_tfidf)
print("Report on train set:",
      classification_report(
          y_train,
          svm_grid_search.predict(train_tfidf)
      ), sep="\n")
print("Reports on dev set:", classification_report(y_dev, svm_pred_dev), sep="\n")


# In[76]:


svm_grid_search.best_params_


# In[81]:


random_forest_grid_search = GridSearchCV(
    dict_models["random_forest"],
    {
        "n_estimators": (10, 20, 25, 50, 75, 100, 125),
        "max_depth": (5, 10, 25, 50),
    }
)
random_forest_grid_search.fit(train_tfidf, y_train)
random_forest_pred = random_forest_grid_search.predict(dev_tfidf)
print("Reports on train set:",
      classification_report(
          y_train,
          random_forest_grid_search.predict(train_tfidf)
      ),
      sep="\n"
     )
print("Reports on dev set:", classification_report(y_dev, random_forest_pred), sep="\n")


# In[ ]:


random_forest_grid_search.best_params_


# In[ ]:


model = Pipeline([
    ("vectorizer", TfidfVectorizer(
        min_df=.015,
        max_df=.7,
        ngram_range=(1, 1),
    )),
    ("model", RandomForestClassifier(
        max_depth=50,
        n_estimators=75,
        random_state=11
    ))
     ])
    # training
    model.fit(X_train, y_train)
    
    pred = model.predict(X_dev)
    print("Reports on train set:",
          classification_report(
              y_train,
              model.predict(X_train)
          ),
          sep="\n"
         )
    print("Reports on dev set:", classification_report(y_dev, pred), sep="\n")


# In[ ]:


joblib.dump(model, "D:/PTA 2016-2017/Modul/dataset/modelyt.joblib")


# In[ ]:


model = joblib.load("D:/PTA 2016-2017/Modul/dataset/modelyt.joblib")
print(model.get_params())


# In[ ]:


preds = model.predict(X_dev)
print(classification_report(y_dev, preds))

