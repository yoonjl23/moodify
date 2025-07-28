import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# 1. 사용자 프로필 로딩 및 필터링
profile_df = pd.read_csv("userid-profile.tsv", sep="\t")
profile_df.rename(columns={"#id": "user_id"}, inplace=True)
profile_df["age"] = pd.to_numeric(profile_df["age"], errors="coerce")

# 조건: 20대 여성, United States
target_users = profile_df[
    (profile_df["gender"] == "f") &
    (profile_df["age"] >= 20) & (profile_df["age"] < 30) &
    (profile_df["country"] == "United States")
]["user_id"].dropna().unique().tolist()

print(f"대상 유저 수: {len(target_users)}")

# 2. 대상 사용자만 필터링해서 로그와 메타데이터 로딩
filtered_chunks = []
track_meta_chunks = []
target_user_set = set(target_users)
chunk_size = 100_000

for chunk in pd.read_csv(
    "userid-timestamp-artid-artname-traid-traname.tsv",
    sep="\t",
    header=None,
    names=["user_id", "timestamp", "artid", "artname", "traid", "traname"],
    usecols=["user_id", "traid", "traname", "artname"],
    on_bad_lines="skip",
    chunksize=chunk_size
):
    # 트랙 제목 정보 수집
    track_meta_chunks.append(chunk[["traid", "traname", "artname"]])

    # 유저 로그 필터링
    filtered = chunk[chunk["user_id"].isin(target_user_set)]
    filtered = filtered.dropna(subset=["user_id", "traid"])
    filtered_chunks.append(filtered[["user_id", "traid"]])

# 로그 결합
df = pd.concat(filtered_chunks, ignore_index=True)
print(f"필터링된 로그 수: {len(df)}")

# 메타데이터 결합 후 중복 제거
track_meta = pd.concat(track_meta_chunks, ignore_index=True)
track_meta = track_meta.dropna(subset=["traid", "traname"]).drop_duplicates("traid")

# 3. 인코딩
user_encoder = LabelEncoder()
track_encoder = LabelEncoder()

df["user_idx"] = user_encoder.fit_transform(df["user_id"])
df["track_idx"] = track_encoder.fit_transform(df["traid"])

# 🛠️ 오류 방지를 위해 메타데이터도 track_encoder에 맞춰 필터링
track_meta = track_meta[track_meta["traid"].isin(track_encoder.classes_)]
track_meta["track_idx"] = track_encoder.transform(track_meta["traid"])

# 4. 사용자-트랙 매트릭스 생성
user_track_matrix = df.groupby(["user_idx", "track_idx"]).size().unstack(fill_value=0)

# 5. 사용자 유사도 계산
cf_similarity = cosine_similarity(user_track_matrix)
cf_similarity = np.nan_to_num(cf_similarity)

# 6. CF 기반 추천 함수 (track_idx 반환)
def recommend_cf(user_raw_id, top_k=10, neighbor_k=5):
    if user_raw_id not in user_encoder.classes_:
        return []

    user_idx = user_encoder.transform([user_raw_id])[0]
    sim_scores = cf_similarity[user_idx]

    # 자기 자신 제외
    sim_scores[user_idx] = 0
    top_neighbors = sim_scores.argsort()[::-1][:neighbor_k]

    # 이웃들의 트랙 집계
    neighbor_matrix = user_track_matrix.iloc[top_neighbors]
    track_scores = neighbor_matrix.sum(axis=0)

    # 사용자가 이미 들은 트랙 제외
    user_listened = set(user_track_matrix.columns[user_track_matrix.iloc[user_idx] > 0])
    track_scores = track_scores.drop(labels=user_listened, errors='ignore')

    # top_k 추천 track_idx 반환
    top_tracks = track_scores.sort_values(ascending=False).head(top_k).index.tolist()
    return top_tracks

# 7. 추천 + 제목 출력 함수
def print_recommendations_with_titles(user_raw_id, top_k=5):
    top_track_indices = recommend_cf(user_raw_id, top_k=top_k)
    
    if not top_track_indices:
        print(f"\n사용자: {user_raw_id} → 해당 사용자 없음 또는 추천 불가")
        return

    print(f"\n사용자: {user_raw_id}")
    for rank, idx in enumerate(top_track_indices, 1):
        match = track_meta[track_meta["track_idx"] == idx]
        if not match.empty:
            row = match.iloc[0]
            print(f"{rank}. 제목: {row['traname']} | 아티스트: {row['artname']}")
        else:
            print(f"{rank}. 🎵 Track Index: {idx} (제목 정보 없음)")

# 8. 추천 실행 (처음 5명)
print("\n추천 결과:") 
for user_id in target_users[:5]:
    print_recommendations_with_titles(user_id, top_k=5)
