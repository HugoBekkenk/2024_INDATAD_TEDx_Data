import os
import requests
import psycopg2
import re
from psycopg2 import sql
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import joblib
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

scaler = joblib.load('scaler.pkl')
kmeans_model = joblib.load('kmeans_model.joblib')
clf = joblib.load('sentiment_model.joblib')
count_vectorizer = joblib.load('vectorizer.joblib')


def gather_environment_variables():
    database = os.getenv('DB_NAME')
    user = os.getenv('DB_USER')
    host = os.getenv('DB_HOST')
    password = os.getenv('DB_PASSWORD')
    api_key = os.getenv('API_KEY')
    port = 5432
    if database is None:
        print("DB_NAME is not set")
    elif user is None:
        print("DB_USER is not set")
    elif host is None:
        print("DB_HOST is not set")
    elif password is None:
        print("DB_PASSWORD is not set")
    elif api_key is None:
        print("API_KEY is not set")
    else:
        environment_info = {"database": database, "user": user, "host": host, "password": password, "api_key": api_key,
                            "port": port}
        return environment_info
    return None


def connect_to_database(environment_info):
    conn = psycopg2.connect(database=environment_info["database"],
                            user=environment_info["user"],
                            host=environment_info["host"],
                            password=environment_info["password"],
                            port=environment_info["port"])
    return conn


def connect_to_youtube_api_gather_items(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        metadata = response.json()

        if "items" in metadata and metadata["items"]:
            items = metadata["items"]
            return items

        else:
            print(f"No metadata found")

    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve metadata: {e}")
    except Exception as e:
        print(f"An error occurred while processing: {e}")

def gather_video_ids():
    video_dir = '/video-container'
    video_ids = []
    if os.path.exists(video_dir):
        for video_file in os.listdir(video_dir):
            video_ids.append(video_file)
        return video_ids
    else:
        print(f"Directory {video_dir} does not exist.")
        return None


def gather_category_info(api_key):
    url = f"https://www.googleapis.com/youtube/v3/videoCategories?part=snippet&regionCode=US&key={api_key}"
    items = connect_to_youtube_api_gather_items(url)
    category_lookup_dictionary = {}
    if items:
        for item in items:
            category_id = int(item["id"])
            category_lookup_dictionary.update({category_id: item["snippet"]["title"]})

    return category_lookup_dictionary


def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([entry['text'].replace("\n", " ") for entry in transcript_list])
        return full_transcript

    except NoTranscriptFound:
        full_transcript = "Transcript not found"
        return full_transcript


def insert_transcript(cur, video_id, transcript):
    sql_transcript_query = sql.SQL("""
                                    INSERT INTO transcript(video_id, transcript)
                                    VALUES (%s, %s)
                                """)
    cur.execute(sql_transcript_query, (video_id, transcript))


def insert_deploy_log(cur):
    cur.execute("""
                                INSERT INTO deploy(deploy_timestamp)
                                VALUES (CURRENT_TIMESTAMP);
                                """)


def parse_duration(duration):
    pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.match(duration)

    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0

    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


def calculate_popularity(item):
    views = int(item["statistics"].get("viewCount", 0))
    likes = int(item["statistics"].get("likeCount", 0))
    comment_count = int(item["statistics"].get("commentCount", 0))

    likes_to_views = likes / views if views > 0 else 0

    feature_data = np.array([[views, likes, comment_count, likes_to_views]], dtype=float)

    scaled_data = scaler.transform(feature_data)

    predicted_cluster = kmeans_model.predict(scaled_data)
    popularity = "not popular" if predicted_cluster[0] == 1 else "popular"
    return popularity


def calculate_sentiment(transcript):
    new_transcripts_bow = count_vectorizer.transform([transcript])

    # Predict sentiment (0 = Negative, 1 = Positive)
    prediction = clf.predict(new_transcripts_bow)

    sentiment = 'Negative' if prediction[0] == 1 else 'Positive'
    return sentiment

def gather_and_trim_rows_from_video_table(cur):
    cur.execute("""
                                SELECT id FROM video
                                """)
    rows = cur.fetchall()

    trimmed_rows = []
    for row in rows:
        row_string = str(row)
        trimmed_row = row_string.replace("(", "").replace(")", "").replace("'", "").replace(",", "")
        trimmed_rows.append(trimmed_row)

    return trimmed_rows


def insert_new_video_data(cur, video_id, trimmed_rows, item, category_lookup_dictionary):
    if video_id not in trimmed_rows:
        title = item["snippet"]["title"].replace("'", "")
        description = item["snippet"]["description"].replace("'", "")
        publish_date = item["snippet"]["publishedAt"]
        duration = item["contentDetails"]["duration"]
        duration_seconds = parse_duration(duration)
        category_id = int(item["snippet"]["categoryId"])
        popularity = calculate_popularity(item)
        transcript = get_transcript(video_id)
        sentiment = calculate_sentiment(transcript)
        if category_id in category_lookup_dictionary.keys():
            category = category_lookup_dictionary[category_id]
        else:
            category = "invalid"

        sql_query = sql.SQL("""
                INSERT INTO video(id, title, description, publish_date, duration_sec, category_id, category, popularity, sentiment)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """)
        cur.execute(sql_query, (
        video_id, title, description, publish_date, duration_seconds, category_id, category, popularity, sentiment))

        insert_transcript(cur, video_id, transcript)


def insert_video_stats(cur, video_id, item):
    views = item["statistics"].get("viewCount", 0)
    likes = item["statistics"].get("likeCount", 0)
    comments = item["statistics"].get("commentCount", 0)

    sql_stats_query = sql.SQL("""
                            INSERT INTO video_stats(video_id, views, likes, comment_amount, measured_time)
                            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                        """)
    cur.execute(sql_stats_query, (video_id, views, likes, comments))


def gather_meta_data_per_video_and_insert(cur, video_ids, trimmed_rows, api_key, category_lookup_dictionary):
    for video_id in video_ids:
        url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&part=snippet,contentDetails,statistics&key={api_key}"
        items = connect_to_youtube_api_gather_items(url)
        item = items[0]
        if item:
            insert_new_video_data(cur, video_id, trimmed_rows, item, category_lookup_dictionary)

            insert_video_stats(cur, video_id, item)


def main():
    video_ids = gather_video_ids()
    environment_info = gather_environment_variables()
    if environment_info and video_ids:

        cur = None
        conn = None

        try:
            conn = connect_to_database(environment_info)
            cur = conn.cursor()

            insert_deploy_log(cur)
            trimmed_rows = gather_and_trim_rows_from_video_table(cur)

            category_lookup_dictionary = gather_category_info(environment_info['api_key'])
            gather_meta_data_per_video_and_insert(cur, video_ids, trimmed_rows, environment_info['api_key'], category_lookup_dictionary)

            conn.commit()

        except psycopg2.Error as db_err:
            print(f"Database error: {db_err}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()


if __name__ == '__main__':
    main()