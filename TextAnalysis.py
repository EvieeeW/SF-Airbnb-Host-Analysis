import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english'))  # use NLTK's stopwords for English
names = [name.lower() for name in df_listing['name']]
names_count = []
for name in names:
    name = name.translate(str.maketrans('', '', string.punctuation))
    words = name.split()
    for word in words:
        if word not in stop_words:
            names_count.append(word)
result = Counter(names_count).most_common()

# Top 20 hot words
top_20 = result[0:20]
top_20_words = pd.DataFrame(top_20, columns=["words","count"])
words = [pair[0] for pair in top_20]
counts = [pair[1] for pair in top_20]
plt.bar(words, counts)
plt.xticks(rotation=90)
plt.xlabel('Words')
plt.ylabel('Count')
plt.title('Top 20 Words')
plt.show()

# Wordclouds
from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in df_listing.name)
wordcloud = WordCloud(max_words=200,background_color="white").generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Sentiment Analysis
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

scores = defaultdict(int)
for word, count in result:
    score = analyzer.polarity_scores(word)
    scores['positive'] += score['pos'] * count
    scores['negative'] += score['neg'] * count
    scores['neutral'] += score['neu'] * count

labels = ['positive', 'negative', 'neutral']
scores_list = [scores['positive'], scores['negative'], scores['neutral']]

plt.bar(labels, scores_list)
plt.xlabel('Sentiment')
plt.ylabel('Score')
plt.title('Sentiment Analysis')
plt.show()

# hotwords by sentiments
scores_by_word = {}
for word, count in result:
    scores_by_word[word] = analyzer.polarity_scores(word)['compound']
sorted_words_by_score = sorted(scores_by_word.items(), key=lambda x: x[1], reverse=True)
top_50_positive_words = [word for word, score in sorted_words_by_score if score > 0][:50]
words_str = " ".join(top_50_positive_words)
wordcloud = WordCloud(background_color="white").generate(words_str)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# Hotwords by the part of speech
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
import string
import pandas as pd

stop_words = set(stopwords.words('english')) | set(['w'])
names = [name.lower() for name in df_listing['name']]
names_count = []
for name in names:
    name = name.translate(str.maketrans('', '', string.punctuation))
    words = name.split()
    for word in words:
        if word not in stop_words:
            names_count.append(word)

tagged_words = nltk.pos_tag(names_count)

pos_dict = defaultdict(lambda: defaultdict(int))
for word, pos in tagged_words:
    pos_dict[pos][word] += 1

top_words = pd.DataFrame(columns=['POS', 'Word', 'Count'])
for pos, words in pos_dict.items():
    sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
    for i in range(min(len(sorted_words), 20)):
        word, count = sorted_words[i]
        if count > 10:
            top_words = top_words.append({'POS': pos, 'Word': word, 'Count': count}, ignore_index=True)
pos_list = ['NN', 'JJ']
top_n = 10
df_top_words = top_words[top_words['POS'].isin(pos_list)].groupby('POS').head(top_n)
sns.set_style('whitegrid')
sns.catplot(x='Count', y='Word', col='POS', data=df_top_words, kind='bar', color='b', aspect=.6)
