"""
This module contains the functions for plotting and visualizing the results.

Here is a list of the functions:
    (a) plot_pairwise_alignment
    (b) plot_similarity_matrix
    (c) plot_similarity_matrix_heatmap
"""

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt

# Plotly
import plotly.graph_objects as go
import plotly.express as px

# Other necessary packages
import numpy as np
import torch
from typing import List, Union, Tuple, Optional
Coordinate = Union[int, float]

# Plot the pairwise alignment between two strings (or lists of strings)
def plot_pairwise_alignment(
    seq1_pieces: Union[str, List[Union[str, int, float]], np.ndarray],
    seq2_pieces: Union[str, List[Union[str, int, float]], np.ndarray],
    alignment: List[Tuple[int, int]] = [],
    str2colordict: Optional[dict] = None,
    padding_factor: float = 1.4,
    linewidth: float = 1.5,
    border_to_box: float = 0.2,
    title: str = 'Pairwise Alignment',
    seq1_name: str = 'Seq 1',
    seq2_name: str = 'Seq 2',
    show: bool = True,
    save: bool = False,
    save_path: str = 'pairwise_alignment.png',
    save_dpi: int = 300,
    save_bbox_inches: str = 'tight',
):
    """
    Plots the pairwise alignment between two strings.

    Arguments:
        seq1_pieces (Union[str, List[Union[str, int, float], np.ndarray]]): The pieces of the first string or list of strings.
        seq2_pieces (Union[str, List[Union[str, int, float], np.ndarray]]): The pieces of the second string or list of strings.
        alignment (List[Tuple[int, int]]): The pairwise alignment between the two strings.
        str2colordict: Optional[dict] = None: A dictionary of colors for each character/string in the union of the two strings.
        padding_factor (float, optional): The factor to use for the padding (default is 1.4).
        linewidth (float, optional): The linewidth to use for the alignment (default is 1.5).
        border_to_box (float, optional): The gap between the border and the box (default is 0.2).
        title (str, optional): The title of the plot (default is 'Pairwise Alignment').
        seq1_name (str, optional): The name of the first sequence (default is 'Seq 1').
        seq2_name (str, optional): The name of the second sequence (default is 'Seq 2').
        show (bool, optional): Whether to show the plot (default is True).
        save (bool, optional): Whether to save the plot (default is False).
        save_path (str, optional): The path to save the plot (default is 'pairwise_alignment.png').
        save_dpi (int, optional): The dpi to use for the plot (default is 300).
        save_bbox_inches (str, optional): The bbox_inches to use for the plot (default is 'tight').

    Returns:
        None

    Remarks:
        The pairwise alignment is a list of tuples of the form (i, j) where i is the index of the character in the first string and j is the index of the character in the second string.
    """
    # Raise an error if str1 and seq2_pieces are not of the same type
    if type(seq1_pieces) != type(seq2_pieces):
        raise TypeError('seq1_pieces and seq2_pieces must be of the same type.')

    # Raise an error if save is True and save_path is None
    if save and save_path is None:
        raise ValueError('Save path is not specified.')

    # Get the length of the strings
    len1 = len(seq1_pieces)
    len2 = len(seq2_pieces)

    # Get the maximum length
    max_len = max(len1, len2)

    # Get the maximum length of the elements in the strings str1 and seq2_pieces
    max_len_chr1 = max([len(str(x)) for x in seq1_pieces])
    max_len_chr2 = max([len(str(x)) for x in seq2_pieces])
    max_len_chr = max(max_len_chr1, max_len_chr2)

    if max_len_chr > 20:
        raise ValueError('The maximum length of the characters in the strings must be less than 20.')

    # Get the scaling factor
    factor = 0.5 + (max_len_chr // 10.0) * 0.5

    # Get the x and y coordinates of the characters
    x_char = np.concatenate((np.arange(len1), np.arange(len2)))
    y_char = np.concatenate((np.zeros(len1), np.ones(len2)))

    # Get the characters
    chars = np.concatenate((np.array(list(seq1_pieces)), np.array(list(seq2_pieces))))

    # Create the figure
    _, ax = plt.subplots(figsize=(2  * max_len * factor, 2 * 2))

    # Get the alignment
    alignment = np.array(alignment)

    # Check if the alignment is not None and not empty
    if len(alignment) > 0:
        indices1 = alignment[:, 0]
        indices2 = alignment[:, 1]
        # Draw the alignment
        for i in range(len(indices1)):
            ax.plot([indices1[i], indices2[i]], [border_to_box, 1.-border_to_box], 'o-',  color='#336699', linewidth=0.75, zorder=2)


    # Draw the characters/strings
    for i, char in enumerate(chars):
        # Get the color of the character if it is in the dictionary
        strip_char = char.strip()
        fc_color = str2colordict[strip_char] if (str2colordict is not None and strip_char in str2colordict) else (0.88, 0.94, 1.0, 1.0)
        ax.text(
        x_char[i],
        y_char[i],
        char, size=12, 
        ha='center', va='center',
        bbox=dict(facecolor=fc_color,
                  edgecolor='#336699',
                  linewidth=linewidth,
                  boxstyle=f'square,pad={padding_factor}',
                  alpha=0.99,
                  ))
        

    # Set the limits of the axes
    ax.set_xlim(-0.5, max_len  - 0.5)
    ax.set_ylim(-0.5, 1.5)

    # Set the ticks of the axes
    ax.set_yticks([0, 1])

    # Set the tick labels of the axes
    ax.set_yticklabels([seq1_name, seq2_name], fontsize=12)#, fontweight='bold')

    # Set the title of the axes
    ax.set_title(title, fontsize=14, fontweight='book')

    # Turn off the spines
    ax.spines[:].set_visible(False)

    # Turn off the ticks
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # tight layout
    plt.tight_layout()

    # Show the plot
    if show:
        plt.show()

    # Save the plot
    if save:
        plt.savefig(save_path, dpi=save_dpi, bbox_inches=save_bbox_inches)


# Plot a heatmap
def plot_heatmap(
    data: Union[List[List[Union[str, int, float]]], np.ndarray],
    title: str = 'Heatmap',
    x_label: str = 'X',
    y_label: str = 'Y',
    x_ticks: List[str] = None,
    y_ticks: List[str] = None,
    colorbar_kwargs: dict = None,
    color_threshold: float = None,
    textcolors=("black", "white"),
    valfmt="{x:.1f}",
    legend: bool = False,
    show: bool = True,
    save: bool = False,
    save_path: str = 'heatmap.png',
    save_dpi: int = 300,
    save_bbox_inches: str = 'tight',
    **kwargs) -> None:
    """
    Creates a heatmap.

    Arguments:
        data (Union[List[List[Union[str, int, float]]], np.ndarray]): The data to plot.
        title (str, optional): The title of the plot (default: 'Heatmap').
        x_label (str, optional): The label of the x-axis (default: 'X').
        y_label (str, optional): The label of the y-axis (default: 'Y').
        x_ticks (List[str], optional): The ticks of the x-axis (default: None).
        y_ticks (List[str], optional): The ticks of the y-axis (default: None).
        colorbar_kwargs (dict, optional): The keyword arguments for the colorbar (default: None).
        color_threshold (float, optional): The threshold to use for the color (default: None).
        textcolors (tuple, optional): The colors to use for the text (default: ("black", "white")).
        valfmt (str, optional): The format to use for the values (default: "{x:.1f}").
        legend (bool, optional): Whether to show the legend (default: False).
        show (bool, optional): Whether to show the plot (default: True).
        save (bool, optional): Whether to save the plot (default: False).
        save_path (str, optional): The path to save the plot (default: 'heatmap.png').
        save_dpi (int, optional): The dpi to use for the plot (default: 300).
        save_bbox_inches (str, optional): The bbox_inches to use for the plot (default: 'tight').
        **kwargs: The keyword arguments for the heatmap.
    """
    # Create the figure and axes
    fig, ax = plt.subplots()
    
    # Create the heatmap
    im = ax.imshow(data, **kwargs)

    # Create the colorbar
    if colorbar_kwargs is None:
        colorbar_kwargs = {}

    # Create the colorbar
    if legend:
        ax.figure.colorbar(im, ax=ax, **colorbar_kwargs)

    # Set the x-axis label
    ax.set_xlabel(x_label, fontweight='medium')

    # Set the y-axis label
    ax.set_ylabel(y_label, fontweight='medium')

    # Set the x-axis ticks
    if x_ticks is not None:
        ax.set_xticks(np.arange(len(x_ticks)))
        ax.set_xticklabels(x_ticks)

    # Set the y-axis ticks
    if y_ticks is not None:
        ax.set_yticks(np.arange(len(y_ticks)))
        ax.set_yticklabels(y_ticks)

    # Set the tick parameters of the axes
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor")

    # Turn off the spines
    ax.spines[:].set_visible(False)

    # Turn off the ticks
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    # Set the grid and tick parameters
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.xaxis.set_label_position('top')

    # Color threshold for the heatmap
    if color_threshold is not None:
        color_threshold = im.norm(color_threshold)
    else:
        color_threshold = im.norm(data.max())/2.

    # Text annotations
    kw = dict(horizontalalignment="center",
              verticalalignment="center")

    # Value format for the text annotations
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create text annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > color_threshold)])
            im.axes.text(j, i, valfmt(data[i, j], None), **kw)

    # Set the title
    fig.suptitle(title, fontsize=14, fontweight='semibold')

    # Set the tight layout    
    plt.tight_layout()

    # Show the plot
    if show:
        plt.show()

    # Save the plot
    if save:
        plt.savefig(save_path, dpi=save_dpi, bbox_inches=save_bbox_inches)


def plot_corpus_embeds_with_plotly(
    corpus_embeddings: Union[List[List[Coordinate]], np.ndarray, torch.Tensor],
    corpus_labels: List[str],
    corpus_hover_texts: List[str],
    corpus_scatter_kwargs: Optional[dict] = {},
    layoot_dict: Optional[dict] = None,
    query_embeddings: Optional[Union[List[List[Coordinate]], np.ndarray]] = None,
    query_labels: Optional[List[str]] = None,
    query_hover_texts: List[str] = None,
    query_modes: Optional[Union[List[str], str]] = 'markers',
    query_marker_dict: Optional[dict] = None,
    show_plot: bool = True,
    save_path: Optional[str] = None,
    ) -> go.Figure:
    """
    This function takes a corpus of embeddings and corresponding labels and plots them in a 2D scatter plot using plotly.
    It also optionally takes a query embedding and corresponding label and plots it in a different color and shape.

    Arguments:
        corpus_embeddings: A list of lists or a numpy array or a torch tensor of corpus embeddings (e.g. sentence embeddings).
        corpus_labels: A list of labels for the corpus embeddings.
        corpus_hover_texts: A list of hover texts for the corpus embeddings.
        corpus_scatter_kwargs: A dictionary of keyword arguments for the corpus scatter plot (e.g. marker size, marker color, etc.) (default: {}).
        layoot_dict: A dictionary of keyword arguments for the layout of the plot (e.g. title, x-axis title, y-axis title, etc.) (default: None).
        query_embeddings: A list of lists or a numpy array of query embeddings (e.g. sentence embeddings) (default: None).
        query_labels: A list of labels for the query embeddings (default: None).
        query_hover_texts: A list of hover texts for the query embeddings (default: None).
        query_modes: A list of modes for the query embeddings (default: 'markers').
        query_marker_dict: A dictionary of keyword arguments for the query scatter plot (e.g. marker size, marker color, etc.) (default: None).
        show_plot: A boolean whether to show the plot (default: True).
        save_path: A string of the path to save the plot (e.g., 'corpus_embeddings.html') (default: None).

    Returns:
        go.Figure: A plotly figure object.

    .. note::
        Please refer to the Hands-on Tutorial on Semantic Search with HUPD Patent Data for a good demonstration of how to use this function.
    """

    # If the corpus_embeddings are a torch tensor or a list, we convert them to a numpy array
    if isinstance(corpus_embeddings, torch.Tensor):
        corpus_embeddings = corpus_embeddings.detach().cpu().numpy()
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = np.array(corpus_embeddings)

    # Let us plot the corpus embeddings
    fig = px.scatter(corpus_embeddings, x=0, y=1, color=corpus_labels, hover_name=corpus_hover_texts, **corpus_scatter_kwargs)

    # If we have query embeddings, we plot them as well
    if query_embeddings is not None:
        # If the query_embeddings are a torch tensor or a list, we convert them to a numpy array
        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.detach().cpu().numpy()
        elif isinstance(query_embeddings, list):
            query_embeddings = np.array(query_embeddings)

        # Check if markers are specified for the query embeddings
        q_marker_dict = query_marker_dict if query_marker_dict is not None else dict(size=10, color='black')

        # If the query_modes is a string, we convert it to a list of the same length as the query_embeddings
        if isinstance(query_modes, str):
            query_modes = [query_modes] * len(query_embeddings)

        # Let us plot the query embeddings on top of the corpus embeddings, one by one
        for i, query_embedding in enumerate(query_embeddings):
            q_mode = query_modes[i] if query_modes is not None else 'markers'
            q_label = query_labels[i] if query_labels is not None else 'Query'
            q_hover_text = query_hover_texts[i] if query_hover_texts is not None else 'Query'

            fig.add_trace(go.Scatter(x=[query_embedding[0]], y=[query_embedding[1]], mode=q_mode, marker=q_marker_dict, name=q_label, hovertext=q_hover_text))

    # If we have a layout dictionary, we update the figure layout with it
    if layoot_dict is not None:
        fig.update_layout(layoot_dict)

    # If we want to save the plot, we do it here
    if save_path is not None:
        fig.write_html(save_path)

    # If we want to show the plot, we do it here
    if show_plot:
        fig.show()

    return fig