import pandas as pd
import re
import json

df = pd.read_csv("data/conversations.csv", header=None)

all_messages = []
message_id = 1

for conv_id, row in df.iterrows():

    conversation = str(row[0])

    pattern = r"(User [12]):\s*(.*?)(?=(?:User [12]:)|$)"

    matches = re.findall(pattern, conversation, re.DOTALL)

    for speaker, text in matches:

        all_messages.append({
            "message_id": message_id,
            "conversation_id": int(conv_id),
            "speaker": speaker,
            "text": text.strip()
        })

        message_id += 1

print("Total Conversations:", len(df))
print("Total Messages:", len(all_messages))

print("\nFirst 5 Messages:\n")

for msg in all_messages[:5]:
    print(msg)

with open("storage/messages.json", "w", encoding="utf-8") as f:
    json.dump(all_messages, f, indent=2, ensure_ascii=False)

print("\nmessages.json saved successfully")