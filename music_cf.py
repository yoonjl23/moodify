import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# ✅ 0. 사용자 입력
gender_input = input("성별을 입력하세요 (f/m): ").strip().lower()
age_input = int(input("나이를 입력하세요: ").strip())
country_input = input("국가명을 입력하세요 (예: United Kingdom): ").strip()

# 1. 사용자 프로필 로딩 및 필터링
profile_df = pd.read_csv("userid-profile.tsv", sep="\t")
profile_df.rename(columns={"#id": "user_id"}, inplace=True)
profile_df["age"] = pd.to_numeric(profile_df["age"], errors="coerce")

# ✅ 입력 기준으로 필터링
target_users = profile_df[
    (profile_df["gender"] == gender_input) &
    (profile_df["age"] >= age_input) & (profile_df["age"] < age_input + 10) &
    (profile_df["country"] == country_input)
]["user_id"].dropna().unique().tolist()

print(f"\n🎯 대상 유저 수: {len(target_users)}")

if not target_users:
    print("조건에 맞는 사용자가 없습니다.")
    exit()

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

# 메타데이터 결합 후 중복 제거
track_meta = pd.concat(track_meta_chunks, ignore_index=True)
track_meta = track_meta.dropna(subset=["traid", "traname"]).drop_duplicates("traid")

# 3. 인코딩
user_encoder = LabelEncoder()
track_encoder = LabelEncoder()

df["user_idx"] = user_encoder.fit_transform(df["user_id"])
df["track_idx"] = track_encoder.fit_transform(df["traid"])

# 오류 방지를 위해 메타데이터도 track_encoder에 맞춰 필터링
track_meta = track_meta[track_meta["traid"].isin(track_encoder.classes_)]
track_meta["track_idx"] = track_encoder.transform(track_meta["traid"])

# 4. 사용자-트랙 매트릭스 생성
user_track_matrix = df.groupby(["user_idx", "track_idx"]).size().unstack(fill_value=0)

# 5. 사용자 유사도 계산
cf_similarity = cosine_similarity(user_track_matrix)
cf_similarity = np.nan_to_num(cf_similarity)

# 6. CF 기반 추천 함수 (track_idx 반환)
def recommend_cf_for_group(user_raw_ids, top_k=5, neighbor_k=5):
    # 유효한 유저만 필터
    valid_ids = [uid for uid in user_raw_ids if uid in user_encoder.classes_]
    if not valid_ids:
        return []

    total_track_scores = pd.Series(dtype=float)

    for user_raw_id in valid_ids:
        user_idx = user_encoder.transform([user_raw_id])[0]
        sim_scores = cf_similarity[user_idx]
        sim_scores[user_idx] = 0
        top_neighbors = sim_scores.argsort()[::-1][:neighbor_k]

        # 이웃들의 트랙 집계
        neighbor_matrix = user_track_matrix.iloc[top_neighbors]
        track_scores = neighbor_matrix.sum(axis=0)

        # 이미 들은 곡 제외
        user_listened = set(user_track_matrix.columns[user_track_matrix.iloc[user_idx] > 0])
        track_scores = track_scores.drop(labels=user_listened, errors='ignore')

        # track_score 누적 합산
        total_track_scores = total_track_scores.add(track_scores, fill_value=0)

    if total_track_scores.empty:
        return []

    # 상위 top_k 트랙 반환
    return total_track_scores.sort_values(ascending=False).head(top_k).index.tolist()


# 7. 추천 + 제목 출력 함수
def print_group_recommendations(user_ids, top_k=5):
    top_track_indices = recommend_cf_for_group(user_ids, top_k=top_k)

    if not top_track_indices:
        print("\n❌ 추천할 곡이 없습니다.")
        return

    print(f"\n🎧 그룹 추천 결과 (유저 수: {len(user_ids)}명):")
    for rank, idx in enumerate(top_track_indices, 1):
        match = track_meta[track_meta["track_idx"] == idx]
        if not match.empty:
            row = match.iloc[0]
            print(f"{rank}. 제목: {row['traname']} | 아티스트: {row['artname']}")
        else:
            print(f"{rank}. Track Index: {idx} (제목 정보 없음)")

# 8. 추천 실행 (상위 5명만 출력)
print("\n📌 추천 결과:")
print_group_recommendations(target_users, top_k=5)
