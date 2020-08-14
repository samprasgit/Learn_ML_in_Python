
train = ["我 爱 北 京 天 安 门", "我 喜 欢 上 海"]
train_bigarm = []
for value in train:
    lentext = len(value.split())
    value = value.split()
    for i in range(lentext - 1):
        value.append("".join(value[i:i + 2]))

    train_bigarm.append(" ".join(value))
print(train_bigarm)
