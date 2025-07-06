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

# Test different similarity thresholds
similarity_thresholds = [0.3, 0.5, 0.7]
results = {}

for threshold in similarity_thresholds:
    recommendations = {}
    for user_id, row in user_course.iterrows():
        seen = set(row.dropna().index)
        # Consider only courses the user rated 4 or 5 (highly rated)
        high_rated = row[row >= 4].dropna().index
        if len(high_rated) == 0:
            continue
        # Aggregate similarity scores for all unseen courses
        sim_scores = pd.Series(0, index=courses_genre.index)
        for course in high_rated:
            sim_scores += course_sim_matrix[course]
        sim_scores = sim_scores.drop(labels=seen, errors='ignore')
        # Apply similarity threshold
        sim_scores = sim_scores[sim_scores >= threshold]
        top_courses = sim_scores.nlargest(5).index.tolist()  # Top 5 within threshold
        recommendations[user_id] = top_courses
    
    # Calculate average recommendations per user
    avg_recommendations = np.mean([len(v) for v in recommendations.values()])
    results[threshold] = {
        'recommendations': recommendations,
        'avg_recommendations': avg_recommendations
    }
    
    # Get top 10 most frequently recommended courses for this threshold
    top_courses_flat = [c for recs in recommendations.values() for c in recs]
    top_10 = Counter(top_courses_flat).most_common(10)
    
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
    
    save_top10_png(top_10, courses, f'top10_recommended_courses_threshold_{threshold}.png')

# Save summary of average recommendations per threshold
def save_summary_png(results, filename):
    fig, ax = plt.subplots(figsize=(8, 4))
    thresholds = list(results.keys())
    avg_recs = [results[t]['avg_recommendations'] for t in thresholds]
    
    bars = ax.bar(thresholds, avg_recs, color='#A3C9A8')
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Average Recommendations per User')
    ax.set_title('Average New Courses Recommended per User by Similarity Threshold')
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_recs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

save_summary_png(results, 'similarity_threshold_analysis.png')

# Print summary
print("Similarity Threshold Analysis:")
for threshold in similarity_thresholds:
    print(f"Threshold {threshold}: Average {results[threshold]['avg_recommendations']:.2f} recommendations per user") 