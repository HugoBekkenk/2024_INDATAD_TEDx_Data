import requests
import psycopg2
import re
import joblib
import numpy as np
import warnings
from psycopg2 import sql
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

scaler = joblib.load('scaler.pkl')
kmeans_model = joblib.load('kmeans_model.joblib')

sentiment_model = joblib.load('sentiment_model.joblib')
nlp_vectorizer = joblib.load('vectorizer.joblib')


video_ids = ["04PmEJaYKd0", "4wtz1AP3y_o", "9_O7qXqrMfE", "cHKs2aVxOmQ", "gHvXr6NuqYQ",
             "kIFOiF9Q4VM", "nCmlX23tmHg", "qEJ4hkpQW8E", "sUv353ua7E8", "0BSaphO1v-U",
             "54AYOd5S7uo", "9WeAOBUdqKM", "CV1yPkkUFOw", "G-lWWxJPfFo", "KKNCiRWd_j0",
             "nv9WwHpOKEg", "QOCZYRXL0AQ", "sxjzLv-DThs", "0G2U0R0hOCU", "58rMMTJUp_U",
             "af_ZpYOk7-Y", "D1m1SQIdMkk", "Hfejyq5nrvE", "KNEGWrD08f8", "OAzp-7yY0FQ",
             "QpzykxnCtvM", "T4Rvj_bBxiA", "0juLRi90kRg", "5m9AYbFqpQo", "AW3klebbNYY",
             "DAEuBdObOok", "hn5gytOCVgE", "kSnvSXDN3Dk", "Od4ZZ8vIFOY", "QQxsvUUNMmw",
             "tlWuP7wESZw", "0Mi0miIN6tA", "69hJ9zN4t3g", "a_yYWpiC3t0", "DAkZpaYa4ZM",
             "HrCbXNRP7eg", "L0L_D2VPXMU", "Oo2upU6ny-I", "_QrZvwNE510", "wLvGABoTV-s",
             "0_M_syPuFos", "6iqXH9RPK1w", "b0Z9IpTVfUg", "DdlTvyQl5ws", "hwSNbMW6XGY",
             "L61Kbo3y218", "orD3vsEyGdA", "QTau-xHsz80", "yTcn-zE41pc", "0xfJy96HJqo",
             "7c_X-opfRrQ", "B5smctuV7-Q", "DWZh9l8xUtY", "HYnZy2Cx7UM", "Li7PsYiwxVc",
             "P_8l1WAJ2Mw", "r6pItuOoGxc", "1bB86TOKljk", "7Fiaew7nDmE", "bB33OBc-U6A",
             "EjNV6JwlV2s", "icr5ParmRwU", "LUaaGfCCfPE", "p_9eJ3uSP00", "Rcm9u9CdK10",
             "1Ws3w_ZOmhI", "7GN10u6F9m0", "BH8dSEFxzjo", "eucTQXM4ymE", "iKBPrJ-AKRs",
             "LUn8IjZKBPg", "Pj9QnO9rZkE", "SCWFVd_FiAU", "2gOvQIMWbCY", "8efxcrjdffg",
             "bhb0P5GGpys", "Fh4YXJ-cqus", "iMBJrvEwv8s", "mGeGLLsiy44", "pOdIn86ZM1E",
             "SFpCQRZOxVE", "2ShZKR5Uo2I", "8SJi0sHrEI4", "bkCpQk_K_jg", "f-yWJoJltoo",
             "IMC8jmEXHfk", "MHZMQLDr-OA", "PxE0TWcT-kA", "sgFq4ty8wSI", "441nwncPN28",
             "9EBkS2kE7uk", "bM1LXa68oxc", "g3CvsPAF3_0", "J-FzHIQ7SOs", "mz_4QLvz2HM",
             "qC1QlUr5mCE", "SjCrlJyFBiI", "4FT5RYuifwE", "9m2wP2cArrY", "CeUoS2T2hhc",
             "g5j9XKkDo-w", "jhOyN59wZKI", "nBN9zG1JNPg", "qDFNgc8jsrc", "SNHUu7YkNjA"]

def connect_to_database():
    conn = psycopg2.connect(database="locaal_tedx_db",
                            user="postgres",
                            host="localhost",
                            password="postgres",
                            port=5432)
    return conn


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
    # comments_to_views = comment_count / views if views > 0 else 0

    feature_data = np.array([[views, likes, comment_count, likes_to_views]], dtype=float)

    scaled_data = scaler.transform(feature_data)

    predicted_cluster = kmeans_model.predict(scaled_data)
    if predicted_cluster[0] == 1:
        popularity = "popular"

    elif predicted_cluster[0] == 0:
        popularity = "not popular"

    else:
        popularity = "unknown"
    return popularity


def calculate_sentiment(transcript):
    new_transcripts_bow = nlp_vectorizer.transform([transcript])

    # Predict sentiment (0 = Negative, 1 = Positive)
    prediction = sentiment_model.predict(new_transcripts_bow)

    sentiment = 'Negative' if prediction[0] == 1 else 'Positive'
    return sentiment


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
        cur.execute(sql_query, (video_id, title, description, publish_date, duration_seconds, category_id, category, popularity, sentiment))

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


def gather_meta_data_per_video_and_insert(cur, video_ids, trimmed_rows, api_key, category_lookup_dictionary):
    for video_id in video_ids:
        url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&part=snippet,contentDetails,statistics&key={api_key}"
        items = connect_to_youtube_api_gather_items(url)
        item = items[0]
        if item:
            insert_new_video_data(cur, video_id, trimmed_rows, item, category_lookup_dictionary)

            insert_video_stats(cur, video_id, item)


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
        if not transcript_list:
            raise ValueError("Empty transcript received.")
        full_transcript = " ".join([entry['text'].replace("\n", " ") for entry in transcript_list])
        return full_transcript
    except NoTranscriptFound:
        return "Transcript not found"
    except Exception as e:
        print(f"Error fetching transcript for video {video_id}: {e}")
        return "Error fetching transcript"

def insert_transcript(cur, video_id, transcript):
    sql_transcript_query = sql.SQL("""
                                    INSERT INTO transcript(video_id, transcript)
                                    VALUES (%s, %s)
                                """)
    cur.execute(sql_transcript_query, (video_id, transcript))


def main():
    if video_ids:

        cur = None
        conn = None

        try:
            conn = connect_to_database()
            cur = conn.cursor()

            trimmed_rows = gather_and_trim_rows_from_video_table(cur)

            api_key = '' # place api key
            category_lookup_dictionary = gather_category_info(api_key)

            gather_meta_data_per_video_and_insert(cur, video_ids, trimmed_rows, api_key, category_lookup_dictionary)

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