import pandas as pd

train_set_1 = "train.jsonl"
train_set_2 = "train.parquet"

train_df_1 = pd.read_json(train_set_1, lines = True)
train_df_2 = pd.read_parquet(train_set_2, engine="pyarrow")

train_df_1["joined_evidence"] = train_df_1["evidence"].apply(
    lambda ev_list: " ".join([" ".join(item[2:] if len(item) > 2 else item) for item in ev_list])
)

train_df_1_temp = train_df_1[["claim", "joined_evidence", "verifiable", "label"]].rename(
    columns={"joined_evidence": "evidence"}
)
train_df_2_temp = train_df_2[["claim", "evidence", "verifiable", "label"]]
combined_df = pd.concat([train_df_1_temp, train_df_2_temp], ignore_index=True)
combined_df = combined_df.drop_duplicates()

combined_df.to_csv("train_data.csv", index = False)