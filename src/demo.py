
import numpy as np
import pandas as pd
from pathlib import Path
from lib import (
    load_dataset,
    get_topic_model,
    topics_per_class,
    concurrence_heatmap,
)

if __name__ == "__main__":
    dataset = load_dataset(
        dataset_path=Path(".") / "data" / "News_Category_Dataset_v3_subset.jsonl"
    )
    print(f"""The dataset contains {len(dataset)} Huffington Post headlines across {dataset["category"].nunique()} categories ({", ".join(dataset["category"].unique())})""")
    print("\nDataset sample:")
    print(dataset[["headline", "category", "date"]].sample(5))

    print("=" * 50)

    topic_model = get_topic_model(
        text_documents=dataset["headline"],
        serialization_path=Path(".") / "models" / "serialized_model.pickle"
    )

    print(f"""Found {len(topic_model.topic_labels_)} topics""")

    topic_model.reduce_topics(nr_topics=32, docs=dataset["headline"])
    print(f"""Reduced to {len(topic_model.topic_labels_)} topics""")

    dataset["predicted_topic"] = topic_model.topics_
    dataset["topic_prob"] = np.max(topic_model.probabilities_, axis=1)

    print("=" * 50)

    topics_per_class(
        topic_model,
        text_documents=dataset["headline"],
        ground_truth_categories=dataset["category"],
        export_path=str(Path(".") / "output" / "topics_per_class.png"),
    )

    print("=" * 50)

    concurrence_table = pd.crosstab(
        index=dataset["category"], 
        columns=dataset["predicted_topic"],
        normalize="columns",
    ).drop(columns=-1)

    print("For each news category, listing topics that where assigned to documents of that category more than 2/3 of the time:\n")
    for cat, topic_probs in concurrence_table.iterrows():
        print(cat, "was associated with topics")
        for similar_topic_idx in topic_probs[topic_probs > 2/3].index.tolist():
            topic_keywords = [term[0] for term in topic_model.topic_representations_[similar_topic_idx]]
            print("  " + f"""{similar_topic_idx:>2}: {", ".join(topic_keywords)}""")
        print()

    concurrence_heatmap(
        concurrence_table, 
        export_path=Path(".") / "output" / "concurrence_heatmap.png",
    )