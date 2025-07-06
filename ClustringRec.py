import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

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
user_profiles = user_profiles.fillna(0)

# Normalize user profile features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
user_profiles_norm = pd.DataFrame(scaler.fit_transform(user_profiles), index=user_profiles.index, columns=user_profiles.columns)

# Clustering model (25 clusters)
num_clusters = 25
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
user_clusters = kmeans.fit_predict(user_profiles_norm)
user_profiles_norm['cluster'] = user_clusters

# Assign each course to the closest cluster centroid
course_centroids = kmeans.cluster_centers_
course_cluster_labels = np.argmax(np.dot(courses_genre.values, course_centroids.T), axis=1)
courses_genre['cluster'] = course_cluster_labels

# For each user, recommend new/unseen courses from their cluster
recommendations = {}
for user_id, row in user_course.iterrows():
    seen = set(row.dropna().index)
    user_cluster = user_profiles_norm.loc[user_id, 'cluster']
    cluster_courses = set(courses_genre[courses_genre['cluster'] == user_cluster].index)
    new_courses = list(cluster_courses - seen)
    recommendations[user_id] = new_courses

# Average recommendations per user
avg_recs = np.mean([len(v) for v in recommendations.values()]) if recommendations else 0

# Top 10 most frequently recommended courses
top_courses_flat = [c for recs in recommendations.values() for c in recs]
top_10 = Counter(top_courses_flat).most_common(10)

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
    plt.title('Top-10 Most Frequently Recommended Courses (Clustering)', pad=20)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

save_top10_png(top_10, courses, 'top10_recommended_courses_clustering.png')

print(f"Average new courses recommended per user: {avg_recs:.2f}") 