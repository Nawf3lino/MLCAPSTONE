import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.table import Table
from sklearn.metrics.pairwise import cosine_similarity

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Helper to draw a box with text
def draw_box(text, xy, ax, width=3, height=1, color="#A3C9A8"):
    box = FancyBboxPatch((xy[0], xy[1]), width, height, boxstyle="round,pad=0.3", ec="black", fc=color, lw=2)
    ax.add_patch(box)
    ax.text(xy[0]+width/2, xy[1]+height/2, text, ha='center', va='center', fontsize=12)

# Step positions
positions = [ (1, 4), (5, 4), (9, 4), (13, 4) ]
labels = [
    "User Ratings",
    "Build User\nProfile Vector",
    "Compare with\nCourse Genre Vectors",
    "Recommend\nCourses"
]

# Draw boxes
for pos, label in zip(positions, labels):
    draw_box(label, pos, ax)

# Draw arrows
for i in range(len(positions)-1):
    start = positions[i]
    end = positions[i+1]
    ax.annotate('', xy=(end[0], end[1]+0.5), xytext=(start[0]+3, start[1]+0.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))

plt.title("Simple Flowchart: Content-Based Recommender System", fontsize=14, pad=20)
plt.savefig('content_based_recommender_flowchart.png', dpi=300, bbox_inches='tight')
plt.close()

# Load data
ratings = pd.read_csv('ratings.csv')
courses = pd.read_csv('course_genre.csv')

# Prepare genre vectors
genre_cols = courses.columns[2:]
courses_genre = courses[['COURSE_ID'] + list(genre_cols)].set_index('COURSE_ID')

# Prepare user-course matrix
user_course = ratings.pivot_table(index='user', columns='item', values='rating')

# Build user profile vectors (mean of genre vectors of rated courses, weighted by rating)
def build_user_profile(user_row):
    rated_courses = user_row.dropna().index
    ratings_vec = user_row.dropna().values
    if len(rated_courses) == 0:
        return np.zeros(len(genre_cols))
    genre_matrix = courses_genre.loc[courses_genre.index.intersection(rated_courses)].values
    if genre_matrix.shape[0] == 0:
        return np.zeros(len(genre_cols))
    weighted = genre_matrix * ratings_vec[:, None]
    return weighted.sum(axis=0) / (np.sum(ratings_vec) if np.sum(ratings_vec) > 0 else 1)

user_profiles = user_course.apply(build_user_profile, axis=1, result_type='expand')
user_profiles.columns = genre_cols

# Recommend top-N new/unseen courses for each user
N = 5
recommendations = {}
for user_id, profile in user_profiles.iterrows():
    if np.all(profile == 0):
        continue
    seen = set(user_course.loc[user_id].dropna().index)
    unseen_courses = courses_genre.index.difference(seen)
    if len(unseen_courses) == 0:
        continue
    unseen_matrix = courses_genre.loc[unseen_courses].values
    sim_scores = cosine_similarity([profile.values], unseen_matrix)[0]
    top_idx = np.argsort(sim_scores)[-N:][::-1]
    unseen_courses_array = np.array(unseen_courses)
    top_courses = unseen_courses_array[top_idx]
    recommendations[user_id] = list(top_courses)

# 1. Average number of new/unseen courses recommended per user
avg_new_courses = np.mean([len(v) for v in recommendations.values()])

# 2. Top-10 most frequently recommended courses
from collections import Counter
top_courses_flat = [c for recs in recommendations.values() for c in recs]
top_10 = Counter(top_courses_flat).most_common(10)

# Save summary as PNG
def save_summary_png(avg_new_courses, filename):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    ax.text(0.5, 0.5, f"Average new/unseen courses recommended per user: {avg_new_courses:.2f}",
            ha='center', va='center', fontsize=14)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

save_summary_png(avg_new_courses, 'avg_new_courses_per_user.png')

# Save top-10 table as PNG
def save_top10_png(top_10, courses, filename):
    top10_df = pd.DataFrame(top_10, columns=['Course ID', 'Count'])
    top10_df = top10_df.merge(courses[['COURSE_ID', 'TITLE']], left_on='Course ID', right_on='COURSE_ID', how='left')
    top10_df = top10_df[['Course ID', 'TITLE', 'Count']]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    table = ax.table(cellText=top10_df.values, colLabels=top10_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(top10_df.columns))))
    plt.title('Top-10 Most Frequently Recommended Courses', pad=20)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

save_top10_png(top_10, courses, 'top10_recommended_courses.png') 