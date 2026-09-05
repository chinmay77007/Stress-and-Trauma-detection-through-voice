from analysis.text_features import extract_text_features


text = """
I am really scared and worried.
I don't know what is going to happen.
I haven't been able to sleep properly.
I feel helpless and very nervous.
"""


features = extract_text_features(text)


print("\nText:")
print(text)

print("\nFeatures:")

for name, value in features.items():
    print(f"{name:20} {value}")
