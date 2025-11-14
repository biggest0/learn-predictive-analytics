"""
Demo script showing how to use the improved plotting functions.

This script demonstrates the easy-to-use plotting functions from feature_interpretation.graph
that allow you to quickly visualize feature relationships with just a few lines of code.
"""

import pandas as pd
import numpy as np
from util.file_handler import get_csv_dataframe
from feature_creation.impute import convertNAcellsToNum
from feature_interpretation.graph import (
    plot_scatter_vs_target,
    plot_scatter_batch,
    plot_correlation_heatmap,
    plot_correlation_heatmap_batch,
    plot_feature_importance
)


def demo_scatter_plots():
    """
    Demonstrate scatter plot functions.
    """
    print("=== Scatter Plot Demo ===")

    # Load and prepare data
    df = get_csv_dataframe()
    df = convertNAcellsToNum('bathrooms', df, "mean")
    df = convertNAcellsToNum('bedrooms', df, "mean")
    df = convertNAcellsToNum('beds', df, "mean")

    # Example 1: Plot specific features
    features = ['accommodates', 'bedrooms', 'bathrooms']
    print(f"Plotting scatter plots for: {features}")
    fig, axes = plot_scatter_vs_target(df, features, target='price', figsize=(15, 5))

    # Example 2: Plot batches of features automatically
    print("Plotting first batch of features (batch 0):")
    fig2, axes2 = plot_scatter_batch(df, target='price', batch_size=4, batch_index=0)

    print("Plotting second batch of features (batch 1):")
    fig3, axes3 = plot_scatter_batch(df, target='price', batch_size=4, batch_index=1)


def demo_heatmap_plots():
    """
    Demonstrate correlation heatmap functions.
    """
    print("\n=== Correlation Heatmap Demo ===")

    # Load and prepare data
    df = get_csv_dataframe()
    df = convertNAcellsToNum('bathrooms', df, "mean")
    df = convertNAcellsToNum('bedrooms', df, "mean")
    df = convertNAcellsToNum('beds', df, "mean")

    # Get numeric features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'price' in numeric_cols:
        numeric_cols.remove('price')

    # Example 1: Plot specific features
    features = numeric_cols[:8]  # First 8 features
    print(f"Plotting correlation heatmap for: {features}")
    fig, ax = plot_correlation_heatmap(df, features, target='price')

    # Example 2: Plot batches automatically
    print("Plotting first batch of correlations (batch 0):")
    fig2, ax2 = plot_correlation_heatmap_batch(df, target='price', batch_size=6, batch_index=0)


def demo_feature_importance():
    """
    Demonstrate feature importance plotting.
    """
    print("\n=== Feature Importance Demo ===")

    # Load and prepare data
    df = get_csv_dataframe()
    df = convertNAcellsToNum('bathrooms', df, "mean")
    df = convertNAcellsToNum('bedrooms', df, "mean")
    df = convertNAcellsToNum('beds', df, "mean")

    # Calculate correlations
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    price_corr = corr_matrix['price'].drop('price')  # Remove self-correlation

    print("Plotting feature importance based on correlation with price:")
    fig, ax = plot_feature_importance(
        price_corr,
        title="Feature Importance (Correlation with Price)",
        figsize=(12, 8)
    )


def demo_advanced_usage():
    """
    Demonstrate advanced usage with customization options.
    """
    print("\n=== Advanced Usage Demo ===")

    # Load and prepare data
    df = get_csv_dataframe()
    df = convertNAcellsToNum('bathrooms', df, "mean")
    df = convertNAcellsToNum('bedrooms', df, "mean")
    df = convertNAcellsToNum('beds', df, "mean")

    # Custom scatter plot with styling
    features = ['accommodates', 'bedrooms']
    fig, axes = plot_scatter_vs_target(
        df, features, target='price',
        figsize=(12, 5),
        alpha=0.7,
        edgecolor='blue',
        grid=True,
        show_plot=False  # Don't show immediately
    )

    # Customize further if needed
    axes[0].set_title('Customized: Accommodates vs Price', fontsize=16, color='red')
    axes[1].set_title('Customized: Bedrooms vs Price', fontsize=16, color='green')

    # Save to file instead of showing
    fig.savefig('custom_scatter_plots.png', dpi=300, bbox_inches='tight')
    print("Saved customized scatter plots to 'custom_scatter_plots.png'")

    # Custom heatmap
    features = ['accommodates', 'bedrooms', 'bathrooms', 'beds']
    fig2, ax2 = plot_correlation_heatmap(
        df, features, target='price',
        figsize=(6, 4),
        cmap="coolwarm",
        show_plot=False
    )

    # Save heatmap
    fig2.savefig('custom_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved customized heatmap to 'custom_heatmap.png'")

    # Close figures to free memory
    fig.clf()
    fig2.clf()


def main():
    """
    Run all plotting demos.
    """
    print("🎨 Plotting Functions Demo")
    print("==========================")
    print("\nThis demo shows how easy it is to create visualizations with just a few lines of code!")
    print("All functions return matplotlib figure/axes objects for further customization.\n")

    try:
        demo_scatter_plots()
        demo_heatmap_plots()
        demo_feature_importance()
        demo_advanced_usage()

        print("\n✅ All demos completed successfully!")
        print("\n💡 Tips:")
        print("- Set show_plot=False to suppress automatic display")
        print("- Use the returned fig, ax objects for further customization")
        print("- Save plots using fig.savefig() instead of plt.show()")
        print("- Adjust figsize, colors, and other parameters as needed")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
