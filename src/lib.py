import pandas as pd
from bertopic import BERTopic
from matplotlib import pyplot as plt

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    load in jsonl dataset at path
    """
    return pd.read_json(
        path_or_buf=dataset_path,
        lines=True,
    )

def get_topic_model(text_documents, serialization_path: str) -> BERTopic:
    """
    retrieve topic model from local storage if available, otherwise instantiate and train on provided text documents
    """
    if serialization_path.exists():
        model = BERTopic.load(serialization_path)
        print(f"[SUCCESS] loaded serialized topic model from `{serialization_path}`")
    else:
        model = BERTopic(
            language="english",
            nr_topics="auto", # automatically reduce amount of topics using HDBSCAN
            zeroshot_min_similarity=0.8, # minimum similarity between a zero-shot topic and a document for assignment; controls confidence needed to assign topic to a document
            calculate_probabilities=True,
            verbose=True,
        )
        print(f"[SUCCESS] instantiated BERTopic model: {str(model)}")
        model.fit_transform(text_documents)
        print(f"[SUCCESS] fitted model to data")
        model.save(
            serialization_path, 
            serialization="pickle", 
            save_ctfidf=True,
        )
        print(f"[SUCCESS] saved serialized topic model from `{serialization_path}`")

    return model

def topics_per_class(
        model: BERTopic, 
        text_documents, 
        ground_truth_categories, 
        export_path: str
    ):
    """
    export visualization of distribution of topic assignment within each ground truth category label
    """
    model.visualize_topics_per_class(
        topics_per_class=model.topics_per_class(
            text_documents,
            ground_truth_categories,
        ), 
        top_n_topics=7,
        normalize_frequency=True,
        width=1080,
        height=540,
    ).update_traces(visible=True).write_image(
        file=export_path, 
        format="png",
        width=1080, 
        height=540,
        engine="kaleido",
    )
    print(f"[SUCCESS] saved topics per class visualization to `{export_path}`")

def concurrence_heatmap(concurrence_table: pd.DataFrame, export_path: str):
    """
    export heatmap visualization of within-topic distributions of each of the ground truth category labels
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.imshow(
        concurrence_table, 
        cmap="YlGn",
        norm="linear",
        vmin=0,
        vmax=1,
    )

    ax.set_yticks(
        ticks=range(len(concurrence_table.index)),
        labels=list(concurrence_table.index),
    )
    ax.set_xlabel("Topic")
    ax.set_ylabel("Category", rotation=0, y=1, ha="left")

    if len(concurrence_table.columns) < 50:
        ax.set_xticks(ticks=range(len(concurrence_table.columns)))
        for i in range(len(concurrence_table.index)):
            for j in range(len(concurrence_table.columns)):
                ax.text(j, i, concurrence_table.round(2).to_numpy()[i, j], ha="center", va="center", color="#dcdcdc", fontsize=9)

    ax.set_title("(Normalized) Concurrence of News Category and Assigned Topics", y=1, va="bottom")

    plt.savefig(export_path, dpi=300)
    print(f"[SUCCESS] saved concurrence heatmap to `{export_path}`")