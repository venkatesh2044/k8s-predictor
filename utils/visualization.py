import time
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class Visualizer:
    """
    Visualizes metrics and predictions
    """
    def __init__(self, output_dir='./visualizations'):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_node_metrics(self, node_metrics, node_name=None, save=True):
        """
        Plot node metrics
        
        Args:
            node_metrics: DataFrame with node metrics
            node_name: Node name to filter (if None, plot all nodes)
            save: Whether to save the plot to a file
        
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Filter by node name if specified
            if node_name:
                df = node_metrics[node_metrics['node_name'] == node_name].copy()
            else:
                df = node_metrics.copy()
            
            if df.empty:
                self.logger.warning("No data to plot")
                return None
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Create plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Node Metrics{f' - {node_name}' if node_name else ''}")
            
            # Plot CPU usage
            if 'cpu_usage_percent' in df.columns:
                self._plot_metric(
                    ax=axes[0, 0],
                    df=df,
                    x='timestamp',
                    y='cpu_usage_percent',
                    title='CPU Usage',
                    ylabel='Usage (%)',
                    group_by='node_name' if not node_name else None
                )
            
            # Plot memory usage
            if 'memory_usage_percent' in df.columns:
                self._plot_metric(
                    ax=axes[0, 1],
                    df=df,
                    x='timestamp',
                    y='memory_usage_percent',
                    title='Memory Usage',
                    ylabel='Usage (%)',
                    group_by='node_name' if not node_name else None
                )
            
            # Plot disk usage
            if 'disk_usage_percent' in df.columns:
                self._plot_metric(
                    ax=axes[1, 0],
                    df=df,
                    x='timestamp',
                    y='disk_usage_percent',
                    title='Disk Usage',
                    ylabel='Usage (%)',
                    group_by='node_name' if not node_name else None
                )
            
            # Plot network load
            if 'network_load_mbps' in df.columns:
                self._plot_metric(
                    ax=axes[1, 1],
                    df=df,
                    x='timestamp',
                    y='network_load_mbps',
                    title='Network Load',
                    ylabel='Load (Mbps)',
                    group_by='node_name' if not node_name else None
                )
            
            plt.tight_layout()
            
            # Save plot if requested
            if save:
                filename = f"node_metrics{f'_{node_name}' if node_name else ''}.png"
                plt.savefig(os.path.join(self.output_dir, filename))
                self.logger.info(f"Saved node metrics plot to {filename}")
            
            return fig
        
        except Exception as e:
            self.logger.error(f"Failed to plot node metrics: {e}")
            return None
    
    def plot_pod_metrics(self, pod_metrics, pod_name=None, namespace=None, save=True):
        """
        Plot pod metrics
        
        Args:
            pod_metrics: DataFrame with pod metrics
            pod_name: Pod name to filter (if None, plot all pods)
            namespace: Namespace to filter (if None, plot all namespaces)
            save: Whether to save the plot to a file
        
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Filter by pod name and namespace if specified
            df = pod_metrics.copy()
            
            if pod_name:
                df = df[df['pod_name'] == pod_name]
            
            if namespace:
                df = df[df['pod_namespace'] == namespace]
            
            if df.empty:
                self.logger.warning("No data to plot")
                return None
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Create plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Pod Metrics{f' - {pod_name}' if pod_name else ''}{f' - {namespace}' if namespace else ''}")
            
            # Plot CPU usage
            if 'pod_cpu_usage_percent' in df.columns:
                self._plot_metric(
                    ax=axes[0, 0],
                    df=df,
                    x='timestamp',
                    y='pod_cpu_usage_percent',
                    title='CPU Usage',
                    ylabel='Usage (%)',
                    group_by='pod_name' if not pod_name else None
                )
            
            # Plot memory usage
            if 'pod_memory_usage_percent' in df.columns:
                self._plot_metric(
                    ax=axes[0, 1],
                    df=df,
                    x='timestamp',
                    y='pod_memory_usage_percent',
                    title='Memory Usage',
                    ylabel='Usage (%)',
                    group_by='pod_name' if not pod_name else None
                )
            
            # Plot network usage
            if 'pod_network_mbps' in df.columns:
                self._plot_metric(
                    ax=axes[1, 0],
                    df=df,
                    x='timestamp',
                    y='pod_network_mbps',
                    title='Network Usage',
                    ylabel='Usage (Mbps)',
                    group_by='pod_name' if not pod_name else None
                )
            
            # Plot restart count
            if 'restart_count' in df.columns:
                self._plot_metric(
                    ax=axes[1, 1],
                    df=df,
                    x='timestamp',
                    y='restart_count',
                    title='Restart Count',
                    ylabel='Count',
                    group_by='pod_name' if not pod_name else None
                )
            
            plt.tight_layout()
            
            # Save plot if requested
            if save:
                filename = f"pod_metrics{f'_{pod_name}' if pod_name else ''}{f'_{namespace}' if namespace else ''}.png"
                plt.savefig(os.path.join(self.output_dir, filename))
                self.logger.info(f"Saved pod metrics plot to {filename}")
            
            return fig
        
        except Exception as e:
            self.logger.error(f"Failed to plot pod metrics: {e}")
            return None
    
    def plot_predictions(self, predictions, actual=None, save=True):
        """
        Plot predictions vs actual values
        
        Args:
            predictions: DataFrame with predictions
            actual: DataFrame with actual values
            save: Whether to save the plot to a file
        
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Create plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Predictions vs Actual")
            
            metrics = [
                ('cpu', 'CPU Usage'),
                ('memory', 'Memory Usage'),
                ('disk', 'Disk Usage'),
                ('network', 'Network Load')
            ]
            
            for i, (metric, title) in enumerate(metrics):
                ax = axes[i // 2, i % 2]
                ax.set_title(title)
                
                # Plot predictions
                pred_col = f'predicted_{metric}_usage_percent' if metric != 'network' else 'predicted_network_load_mbps'
                if predictions is not None and pred_col in predictions.columns:
                    ax.plot(predictions.index, predictions[pred_col], 'b-', label='Predicted')
                
                # Plot actual values
                actual_col = f'{metric}_usage_percent' if metric != 'network' else 'network_load_mbps'
                if actual is not None and actual_col in actual.columns:
                    ax.plot(actual.index, actual[actual_col], 'r-', label='Actual')
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
            
            plt.tight_layout()
            
            # Save plot if requested
            if save:
                filename = f"predictions_{int(time.time())}.png"
                plt.savefig(os.path.join(self.output_dir, filename))
                self.logger.info(f"Saved predictions plot to {filename}")
            
            return fig
        
        except Exception as e:
            self.logger.error(f"Failed to plot predictions: {e}")
            return None
    
    def _plot_metric(self, ax, df, x, y, title, ylabel, group_by=None):
        """
        Helper method to plot a metric
        
        Args:
            ax: Matplotlib axis
            df: DataFrame
            x: x-axis column
            y: y-axis column
            title: Plot title
            ylabel: y-axis label
            group_by: Column to group by
        """
        ax.set_title(title)
        
        if group_by:
            for group, group_df in df.groupby(group_by):
                ax.plot(group_df[x], group_df[y], label=group)
            ax.legend()
        else:
            ax.plot(df[x], df[y])
        
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        
        # Add threshold line for usage percentages
        if 'usage_percent' in y:
            ax.axhline(y=85, color='r', linestyle='--', alpha=0.5)