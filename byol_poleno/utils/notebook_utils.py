import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def interactive_pca_3d_highlight(df, highlight_column="species", width=800, height=600):
    """Interactive 3D PCA plot with dropdown to highlight different species.

    Args:
        df (pd.DataFrame): DataFrame containing PCA components and species column.
        highlight_column (str, optional): Column name to highlight. Defaults to "species".
    """

    highlight_list = df[highlight_column].unique()
    highlight_color = "red"
    marker_size = 2
    marker_opacity = 0.8

    # --- 3D Scatter (start with normal coloring by species) ---
    fig3d = px.scatter_3d(
        df, x="PC1", y="PC2", z="PC3", color=highlight_column,
        title="First Three Components for: " + highlight_column,
        opacity=marker_opacity
    )
    fig3d.update_traces(marker=dict(size=marker_size, opacity=marker_opacity))

    # --- Build dropdown buttons ---
    buttons = []

    # Default: all species colored
    buttons.append(dict(
        label="Everything",
        method="update",
        args=[{"marker": [dict(color=None, size=marker_size, opacity=marker_opacity) 
                        for _ in fig3d.data]},
            {"legend": {"title": {"text": highlight_column}}}]
    ))

    # Highlight each species
    for sp in highlight_list:
        new_markers = []
        for trace in fig3d.data:
            trace_species = trace.name  # each trace = one species
            if trace_species == sp:
                new_markers.append(dict(color=highlight_color, 
                                        size=marker_size, 
                                        opacity=marker_opacity))
            else:
                new_markers.append(dict(color="lightgray", 
                                        size=marker_size, 
                                        opacity=0.5))
        buttons.append(dict(
            label=f"{sp}",
            method="update",
            args=[{"marker": new_markers},
                {"legend": {"title": {"text": f"{sp}"}}}]
        ))

    # --- Add dropdown ---
    fig3d.update_layout(
        updatemenus=[dict(
            type="dropdown",
            x=1.1, y=1.15,
            buttons=buttons
        )],
        width=width,
        height=height,
        legend=dict(itemsizing="constant"),
        showlegend=False,
        scene=dict(aspectmode="cube")
    )

    fig3d.show()


def show_confusion_matrix(y_true, y_pred, classes, figsize=(10,8), normalize=False, fontsize=8):

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".1f"
    else:
        fmt = "d"
  
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, cmap="Purples",
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": fontsize})
    
    ax.set_xlabel("Predicted", fontsize=fontsize+2)
    ax.set_ylabel("True", fontsize=fontsize+2)
    ax.set_title("Confusion Matrix", fontsize=fontsize+4)
    
    # Ticks
    plt.xticks(fontsize=fontsize, rotation=90, ha="right")
    plt.yticks(fontsize=fontsize)
    
    plt.tight_layout()
    plt.show()