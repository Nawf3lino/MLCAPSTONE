import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Load data
ratings = pd.read_csv('ratings.csv')
courses = pd.read_csv('course_genre.csv')

# Prepare genre vectors
genre_cols = courses.columns[2:]
courses_genre = courses[['COURSE_ID'] + list(genre_cols)].set_index('COURSE_ID')

# Prepare user-course matrix
user_course = ratings.pivot_table(index='user', columns='item', values='rating')

# Compute course-to-course similarity matrix
course_sim_matrix = pd.DataFrame(
    cosine_similarity(courses_genre.values),
    index=courses_genre.index,
    columns=courses_genre.index
)

thresholds = [0.3, 0.7]
results = {}

for threshold in thresholds:
    recommendations = {}
    for user_id, row in user_course.iterrows():
        seen = set(row.dropna().index)
        # Consider only courses the user rated 4 or 5 (enrolled/highly rated)
        enrolled = row[row >= 4].dropna().index
        if len(enrolled) == 0:
            continue
        # Unselected courses
        unselected = set(courses_genre.index) - seen
        user_recs = set()
        for enrolled_course in enrolled:
            if enrolled_course not in course_sim_matrix.index:
                continue
            for unselected_course in unselected:
                sim = course_sim_matrix.loc[enrolled_course, unselected_course]
                if sim > threshold:
                    user_recs.add(unselected_course)
        recommendations[user_id] = list(user_recs)
    # Average recommendations per user
    avg_recs = np.mean([len(v) for v in recommendations.values()]) if recommendations else 0
    # Top 10 most frequently recommended courses
    top_courses_flat = [c for recs in recommendations.values() for c in recs]
    top_10 = Counter(top_courses_flat).most_common(10)
    results[threshold] = {
        'recommendations': recommendations,
        'avg_recs': avg_recs,
        'top_10': top_10
    }
    # Save top 10 table as PNG
    def save_top10_png(top_10, courses, filename):
        top10_df = pd.DataFrame(top_10, columns=['Course ID', 'Count'])
        top10_df = top10_df.merge(courses[['COURSE_ID', 'TITLE']], left_on='Course ID', right_on='COURSE_ID', how='left')
        top10_df = top10_df[['Course ID', 'TITLE', 'Count']]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        table = ax.table(cellText=top10_df.values, colLabels=top10_df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(top10_df.columns))))
        plt.title(f'Top-10 Most Frequently Recommended Courses (Threshold: {threshold})', pad=20)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    save_top10_png(top_10, courses, f'top10_recommended_courses_CS2_threshold_{threshold}.png')

# Print summary
for threshold in thresholds:
    print(f"Threshold {threshold}: Average {results[threshold]['avg_recs']:.2f} recommendations per user") 