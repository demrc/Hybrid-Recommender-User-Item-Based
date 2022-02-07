import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

#User Based Recommendation
#Part 1
import pandas as pd
movie = pd.read_csv('recommender_systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('recommender_systems/datasets/movie_lens_dataset/rating.csv')
def create_user_movie_df():
    df = movie.merge(rating, how="left", on="movieId")
    rate_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = rate_counts[rate_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

#Part 2

random_user = int(pd.Series(user_movie_df.index).sample(1).values)
random_user_df = user_movie_df[user_movie_df.index==random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
len(movies_watched)

#Part 3

movies_watched_df = user_movie_df[movies_watched]
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId","movie_count"]
perc = len(movies_watched)*60/100
users_same_movies = user_movie_count[user_movie_count["movie_count"]>=perc]["userId"]

#Part 4

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],random_user_df[movies_watched]])
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df,columns = ["corr"])
corr_df.index.names = ["user_id_1","user_id_2"]
corr_df = corr_df.reset_index()
top_users = corr_df[(corr_df["user_id_1"]==random_user) & (corr_df["corr"]>=0.65)][["user_id_2","corr"]].reset_index(drop=True)
top_users.rename(columns = {"user_id_2":"userId"},inplace=True)
top_users_ratings = top_users.merge(rating[["userId","movieId","rating"]],how = "inner")
top_user_ratings = top_users_ratings[top_users_ratings["userId"]!=random_user]

#Part 5

top_users_ratings["weighted_rating"]=top_users_ratings["corr"]*top_users_ratings["rating"]
top_users_ratings.groupby("movieId").agg({"weighted_rating":"mean"})
recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating":"mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df[recommendation_df["weighted_rating"]>3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"]>3.5].sort_values("weighted_rating",ascending=False).head()

#movies names

movies_to_be_recommend.merge(movie[["movieId","title"]])

#-----------Item Base Recommend-------------

movie = pd.read_csv('recommender_systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('recommender_systems/datasets/movie_lens_dataset/rating.csv')

df = movie.merge(rating, how="left", on="movieId")
high_rated = df[(df["userId"]==random_user)&(df["rating"]==5)].sort_values("timestamp",ascending=False)["movieId"].iloc[0]

movie_name = movie[movie["movieId"]==high_rated]["title"].values[0]
movie_name = user_movie_df[0:len(movie_name)]
recommend = user_movie_df.corrwith(movie_name).sort_values(ascending=False).head()
