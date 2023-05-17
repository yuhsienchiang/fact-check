import json

train_data = json.load(open("./train-claims.json")) 
dev_data = json.load(open("./dev-claims.json"))

combined_data = {**train_data, ** dev_data}

f_out = open("./train-dev-claims.json", 'w')
json.dump(combined_data, f_out)
f_out.close()
print("Done!")
