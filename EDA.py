import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS

# Load the data
df = pd.read_csv('course_genre.csv')

# Select only the genre columns (skip COURSE_ID and TITLE)
genre_columns = df.columns[2:]

# Count the number of courses per genre
genre_counts = df[genre_columns].sum().sort_values(ascending=False)

# Plot the bar chart
plt.figure(figsize=(10,6))
# Use a minimalist, soft color (e.g., pastel green)
bar = genre_counts.plot(kind='bar', color='#A3C9A8')
plt.xlabel('Genre')
plt.ylabel('Number of Courses')
plt.title('Number of Courses per Genre')
plt.tight_layout()
plt.savefig('course_genre_counts.png', dpi=300)
plt.close()

# Enrollment distribution histogram
ratings_df = pd.read_csv('ratings.csv')
user_enrollments = ratings_df.groupby('user')['item'].count()

plt.figure(figsize=(10,6))
plt.hist(user_enrollments, bins=range(1, user_enrollments.max()+2), color='#A3C9A8', edgecolor='black')
plt.xlabel('Number of Courses Enrolled')
plt.ylabel('Number of Users')
plt.title('Distribution of Course Enrollments per User')
plt.tight_layout()
plt.savefig('user_enrollment_distribution.png', dpi=300)
plt.close()

# List the most popular 20 courses
popular_courses = ratings_df['item'].value_counts().head(20)
print('Top 20 Most Popular Courses:')
print(popular_courses)

# Merge with course_genre.csv to get titles
course_info = pd.read_csv('course_genre.csv')[['COURSE_ID', 'TITLE']]
popular_courses_df = popular_courses.reset_index()
popular_courses_df.columns = ['Course ID', 'Enrollments']
popular_courses_df = popular_courses_df.merge(course_info, left_on='Course ID', right_on='COURSE_ID', how='left')
popular_courses_df = popular_courses_df[['Course ID', 'TITLE', 'Enrollments']]
popular_courses_df.columns = ['Course ID', 'Title', 'Enrolls']

fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('off')
table = ax.table(
    cellText=popular_courses_df.values,
    colLabels=popular_courses_df.columns,
    cellLoc='center',
    loc='center',
    colColours=['#A3C9A8', '#A3C9A8', '#A3C9A8']
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(col=list(range(len(popular_courses_df.columns))))
plt.title('Top 20 Most Popular Courses', pad=20)
plt.savefig('top_20_courses.png', dpi=300, bbox_inches='tight')
plt.close()

# Custom stopwords for the word cloud
custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    'getting started', 'using', 'enabling', 'template', 'university', 'end', 'introduction', 'basic'
])

# Generate a word cloud from course titles
course_titles = pd.read_csv('course_genre.csv')['TITLE']
title_text = ' '.join(course_titles.dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens', stopwords=custom_stopwords, max_words=100).generate(title_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('course_titles_wordcloud.png', dpi=300, bbox_inches='tight')
plt.close() 