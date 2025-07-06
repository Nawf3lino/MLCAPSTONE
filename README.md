# Course Recommendation System

This project implements a comprehensive course recommendation system using multiple approaches, including content-based filtering, clustering, and collaborative filtering (KNN, NMF, Neural Network Embedding).

## Features
- **Content-Based Filtering:** Recommends courses based on user profiles and course genres, as well as course-to-course similarity.
- **Clustering-Based Recommender:** Groups users by their interests and recommends cluster-specific courses.
- **Collaborative Filtering:** Includes KNN, NMF, and neural network embedding models for user-item rating prediction.
- **Evaluation:** Compares models using metrics like RMSE and visualizes results with bar charts and tables.

## Project Structure
- `EDA.py`: Exploratory data analysis and visualizations.
- `URS.py`: User profile content-based recommender.
- `CS.py`, `CS2.py`: Course similarity-based recommenders with threshold tuning.
- `ClustringRec.py`: Clustering-based recommender system.
- `CollabFilter.py`: Collaborative filtering models and performance comparison.
- Data files: `course_genre.csv`, `ratings.csv`.
- Output images: Visualizations and result tables (PNG files).

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (You may also need to install PyTorch for neural network models.)
2. Run the desired Python scripts for analysis and recommendations:
   ```bash
   python EDA.py
   python URS.py
   python CS2.py
   python ClustringRec.py
   python CollabFilter.py
   ```
3. View the generated PNG files for results and visualizations.

## Notes
- The project supports Apple Silicon (M1/M2/M3) for PyTorch with MPS acceleration.
- Adjust hyperparameters in the scripts to experiment with different recommendation strategies.

## License
This project is for educational and research purposes. 