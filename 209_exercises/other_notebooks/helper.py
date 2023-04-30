
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples


def pca_plots(X_std, best_labels, ind_sil):
	fig, axs = plt.subplots(1,2, figsize=(15,5))
	# PCA Plot
	pca = PCA(n_components=2).fit(X_std)
	X_pca = pca.transform(X_std)
	sc = axs[0].scatter(X_pca[:,0], X_pca[:,1], c=best_labels);
	axs[0].set_xlabel(f'PCA 1 {pca.explained_variance_ratio_[0]:.2%}')
	axs[0].set_ylabel(f'PCA 2 {pca.explained_variance_ratio_[1]:.2%}')
	axs[0].legend(*sc.legend_elements(), title='clusters')
	axs[0].set_title('Clustered Wine Data')

	# Silhouette Plot
	# Average sillhouette score for all points clusters
	avg_sil = ind_sil.mean()
	# position on y-axis to begin plotting a cluster's scores
	y_lower = 10
	for label in set(best_labels):
	    # sort scores for points within a given cluster
	    sorted_scores = np.sort(ind_sil[best_labels==label])
	    n_points = sorted_scores.shape[0]
	    # position on y-axis to end plotting a cluster's scores
	    y_upper = y_lower + n_points
	    color = cm.viridis(label/max(set(best_labels)))
	    axs[1].fill_betweenx(np.arange(y_lower, y_upper),
	                      0, sorted_scores,
	                      facecolor=color, edgecolor=color)
	    axs[1].text(-0.2, np.mean([y_lower, y_upper]), f'Cluster {label}')
	    axs[1].set_yticks([])
	    axs[1].set_xlabel('Silhouette Score')
	    y_lower = y_upper + 10
	axs[1].axvline(avg_sil, c='r', ls='--') 
	axs[1].axvline(0, c='k', ls=':', lw=1, alpha=0.75) 
	axs[1].spines['left'].set_visible(False)
	bottom, top = axs[1].set_xlim()
	axs[1].set_xlim(bottom, 1)
	axs[1].set_title(f'Average Sillhouette Score: {avg_sil:.2f}');
