# Recommendation Systems Lesson Script

## Opening Hook

Good morning everyone. Before writing any code today, pause and think about three questions that sit at the heart of modern artificial intelligence systems used by apps every day.

**Question 1:** How does a favorite app decide which video, song, or game to show next?

Think about opening YouTube, TikTok, Netflix, or Spotify. Often, nothing is typed into a search bar, yet the app still presents something that feels strangely relevant. That is not magic. It is the result of data, mathematical modeling, and machine learning systems trained to predict preference.

**Question 2:** Is the most popular thing always the best thing for a specific person?

Suppose a song has been streamed 50 million times. That tells us the song is broadly popular, but it does not guarantee that every individual listener will enjoy it. Popularity is a crowd-level signal. Recommendation is a personal-level prediction.

**Question 3:** What data does an app quietly collect from clicks, watches, pauses, replays, and skips?

Every interaction can become data. A long watch time may act as a positive signal. A fast skip may act as a negative signal. Even repeated listens, scrolling speed, or time of day can help a system infer patterns about a user.

Ask students to jot down initial answers in a notebook. At the end of the lesson, return to these questions and compare their early intuitions with their new understanding.

## The Core Problem

Recommendation systems solve a very practical computer science problem: there are too many choices, and users need helpful predictions quickly.

Imagine a platform with 100,000 items and thousands or millions of users. A user opens the app and expects useful suggestions almost instantly. The system must decide which small set of items is most likely to be relevant.

This problem involves two competing goals:

- **Relevance:** Show something the user is likely to enjoy.
- **Discovery:** Show something valuable the user might not have found alone.

A system that only repeats familiar content becomes narrow and repetitive. A system that only pushes novelty may ignore obvious preferences. Strong recommendation systems balance both.

This tradeoff is often described as **exploitation versus exploration**:

- **Exploitation:** Use what the system already knows about the user.
- **Exploration:** Test new possibilities to learn more and broaden the user experience.

That balance is one reason recommendation systems are such an important area in artificial intelligence.

## Major Approaches

### Popularity-Based Recommendation

The simplest recommender shows what is most popular overall.

In this approach, the system counts how many users liked, viewed, purchased, or interacted with each item, then ranks items by that total. The top items are shown to everyone or to broad groups of users.

This method is easy to build and often works reasonably well for new platforms. However, it has a serious weakness: it ignores individual differences. A globally popular item may still be a poor match for a particular user.

In simple form, the system can assign a score such as:

\[
\text{Score}(i) = \sum_{u \in U} r_{u,i}
\]

Here, \(r_{u,i}\) represents the interaction or rating from user \(u\) for item \(i\). Higher totals produce higher-ranked items.

This method is useful as a baseline, but it is not truly personalized.

### Content-Based Filtering

Content-based filtering focuses on the properties of the items themselves.

For example, a movie can be described by genre, director, actors, release year, mood, pace, and other features. A song can be represented by artist, genre, tempo, or acoustic qualities. A news article can be represented by topic, writing style, or keywords.

If a user consistently likes science fiction films, the system may recommend other films with similar features. In this way, the engine builds a profile of the user's taste and looks for items that resemble that profile.

A common similarity measure is cosine similarity:

\[
\text{sim}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}||\vec{b}|}
\]

This compares how closely two vectors point in the same direction.

The strength of content-based filtering is that it can make personalized recommendations even without relying heavily on other users. The limitation is that it often becomes too narrow. If a user likes only a few types of items, the system may keep recommending near-copies instead of helping the user discover something surprising.

### Collaborative Filtering

Collaborative filtering is one of the most important ideas in recommendation systems.

Its core insight is simple: people with similar behavior often like similar things. Instead of asking, "What is this item made of?" collaborative filtering asks, "Who behaves similarly, and what did they enjoy?"

There are two classic forms:

- **User-based collaborative filtering:** Find users similar to a target user and recommend items those similar users liked.
- **Item-based collaborative filtering:** Find items similar to a target item and recommend them to users who liked related items.

A common prediction formula for user-based collaborative filtering is:

\[
\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N(u)} \text{sim}(u,v) \cdot (r_{v,i} - \bar{r}_v)}{\sum_{v \in N(u)} |\text{sim}(u,v)|}
\]

This equation says that the predicted rating for user \(u\) on item \(i\) depends on how similar nearby users rated that item, adjusted by their usual rating behavior.

This is a major step beyond popularity-based methods because it captures patterns of agreement between users.

## Matrix Factorization

Modern recommendation systems often go further than direct neighborhood methods and use **matrix factorization**.

Imagine a large user-item matrix. Each row is a user. Each column is an item. Each cell contains a rating, click signal, or watch behavior. Most cells are empty because most users interact with only a tiny fraction of all available content.

The challenge is to estimate the missing values.

Matrix factorization assumes that users and items can both be represented in a smaller hidden feature space. These hidden dimensions are called **latent factors**. They are not manually defined. Instead, the system learns them from the data.

The basic idea is:

\[
R \approx P Q^T
\]

Where:

- \(R\) is the original user-item matrix.
- \(P\) is the user-factor matrix.
- \(Q\) is the item-factor matrix.

A predicted rating is then computed by the dot product:

\[
\hat{r}_{u,i} = \vec{p}_u \cdot \vec{q}_i
\]

To learn the matrices, the model minimizes prediction error on known interactions while adding regularization to avoid overfitting:

\[
\min_{P,Q} \sum_{(u,i) \in \text{known}} (r_{u,i} - \vec{p}_u \cdot \vec{q}_i)^2 + \lambda (||\vec{p}_u||^2 + ||\vec{q}_i||^2)
\]

This is one of the most elegant examples of machine learning in practical use. The system learns hidden structure in user preferences and uses that structure to predict future behavior.

## Data Signals

A recommendation system depends on data. Some of that data is explicit, and some is implicit.

### Explicit Signals

These are signals users intentionally provide:

- Likes
- Star ratings
- Reviews
- Favorites or saves
- Shares

### Implicit Signals

These are inferred from behavior:

- Watch duration
- Replay count
- Skips
- Pause behavior
- Scroll speed
- Time of day
- Device type
- Session sequence

Implicit signals are often especially valuable because users do not always rate what they consume, but they always leave behavioral traces.

A short watch may suggest low interest. A full watch followed by a replay may suggest strong interest. Sequence also matters: what a user viewed just before or after an item can add context to the recommendation.

## Ethics and Responsibility

Recommendation systems are not neutral.

They are built by people, trained on data, and optimized according to business goals. Because of that, technical design choices can produce social consequences.

Important concerns include:

- **Consent:** Do users understand what data is being collected and modeled?
- **Bias amplification:** If training data contains bias, the model may learn and reinforce it.
- **Filter bubbles:** Over-personalization can trap users in narrow informational spaces.
- **Extremity bias:** Systems optimized only for engagement may surface sensational or polarizing content.
- **Diversity and serendipity:** Good systems often benefit from deliberately showing some content outside a user's usual pattern.

Students should understand that building intelligent systems also means taking responsibility for their downstream effects.

## Coding Walkthrough

This section demonstrates a simple user-based collaborative filtering system in Python.

### Step 1: Build a Small Ratings Dataset

```python
import numpy as np
import pandas as pd

ratings = {
    'Inception': [5, 3, np.nan, 1, 4],
    'The Notebook': [1, 5, 4, np.nan, 2],
    'Interstellar': [4, np.nan, 2, 1, 5],
    'Pride & Prejudice': [np.nan, 4, 5, 2, 1],
    'The Dark Knight': [5, 2, np.nan, 1, 4],
    'Mamma Mia': [2, 5, 4, np.nan, 1]
}

df = pd.DataFrame(ratings, index=['Alice', 'Bob', 'Carol', 'Dave', 'Eve'])
print(df)
```

Explain that each row is a user and each column is an item. Missing values represent items not yet rated.

### Step 2: Compute Similarity Between Users

```python
from sklearn.metrics.pairwise import cosine_similarity

df_filled = df.apply(lambda row: row.fillna(row.mean()), axis=1)
similarity_matrix = cosine_similarity(df_filled)
similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)
print(similarity_df)
```

This step approximates similarity by comparing user rating patterns. Students should notice that users with similar preferences receive higher similarity scores.

### Step 3: Predict a Missing Rating

```python
def predict_rating(user, item, df, similarity_df, n_neighbors=3):
    similar_users = similarity_df[user].drop(user).sort_values(ascending=False)
    neighbors = []
    for neighbor, sim in similar_users.items():
        if not np.isnan(df.loc[neighbor, item]):
            neighbors.append((neighbor, sim))
        if len(neighbors) == n_neighbors:
            break

    numerator = sum(sim * df.loc[neighbor, item] for neighbor, sim in neighbors)
    denominator = sum(abs(sim) for _, sim in neighbors)
    return numerator / denominator if denominator != 0 else df[item].mean()

print(predict_rating('Alice', 'Pride & Prejudice', df, similarity_df))
```

This function estimates how a target user might rate an unseen item by looking at the weighted opinions of similar users.

### Step 4: Generate Recommendations

```python
def recommend(user, df, similarity_df, top_n=3):
    unseen = df.loc[user][df.loc[user].isna()].index.tolist()
    predictions = {item: predict_rating(user, item, df, similarity_df) for item in unseen}
    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]

print(recommend('Alice', df, similarity_df))
```

This final step ranks unseen items by predicted score and returns the strongest recommendations.

## Teaching Notes

Use the lesson to emphasize that recommendation systems are not only about coding. They combine mathematics, data science, software engineering, and ethics.

A useful classroom flow is:

1. Start with familiar apps and student intuition.
2. Define the recommendation problem clearly.
3. Compare popularity-based, content-based, and collaborative methods.
4. Introduce matrix factorization as the scalable modern idea.
5. Walk through the code slowly and connect each line to the underlying concept.
6. End with ethics, reflection, and discussion.

## Reflection Questions

Close the lesson by returning to the opening questions.

- How does an app decide what to show next?
- Why is popularity not the same as personalization?
- What kinds of data are collected from user behavior?
- What responsibilities come with designing these systems?

The central idea students should leave with is this: recommendation systems are predictive engines that learn from behavior, estimate preference, and shape what people see. That makes them both technically powerful and socially important.
